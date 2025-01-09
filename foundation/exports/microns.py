import os
import torch
import pandas as pd
from tqdm import tqdm
from shutil import make_archive, rmtree
from datajoint import AndList
from collections import OrderedDict

NETWORK_ID = "c17d459afa99a88b3e48a32fbabc21e4"
INSTANCE_ID = "6600970e9cfe7860b80a70375cb6f20c"
DATA_IDS = [
    "232ba7ad384c7b93f58842b51e7e1ef6",
    "98a58d55e28951a38cc61cfec7d63f76",
    "6dea1dbe556674b1ebd8f984edd102c0",
    "45dc6cb6ed757fb223cdfa846060a2bf",
    "c947b82486ab3d4dfd2972e21ad2ce3b",
    "54aa25585c44713e66671a63068cc5a6",
    "748c36efcdb2b94d5a7e587ca3e80005",
    "cf36b881ec9c3e3aa50986a813ea33d5",
    "fa774b612d638ec8332e0a2fbdc2ed59",
    "e75fe99564d314b39e7348e6a9793cc8",
    "71b64f25a1671f02a589b2dfd3ab0f69",
    "96c3fdd9b93a2736615d43fee8d0d037",
    "11e7be67a39d58be8a10202f654af2b3",
]


def export(target_dir=None):
    """
    Parameters
    ----------
    target_dir : str | os.PathLike | None
        target directory

    Returns
    -------
    str
        export file path
    """
    from foundation.fnn.model import Model
    from foundation.fnn.query.scan import VisualScanRecording

    if target_dir is None:
        target_dir = os.getcwd()

    mdir = os.path.join(target_dir, "microns")
    os.makedirs(mdir, exist_ok=False)
    assert not os.path.exists(f"{mdir}.zip"), f"{mdir}.zip already exists"

    dfs = []
    scans = []

    for i, data_id in enumerate(tqdm(DATA_IDS, desc="Scans")):

        # scan meta data
        recording = VisualScanRecording & {"data_id": data_id}

        units = recording.units
        units = units.fetch(
            "session",
            "scan_idx",
            "unit_id",
            "trace_order",
            order_by="trace_order",
            as_dict=True,
        )
        units = pd.DataFrame(units).rename(columns={"trace_order": "readout_id"})
        dfs.append(units)

        session, scan_idx = recording.key.fetch1("session", "scan_idx")
        scan = {
            "session": session,
            "scan_idx": scan_idx,
            "units": len(units),
            "data_id": data_id,
        }
        scans.append(scan)

        # scan model
        params = Model & {
            "data_id": data_id,
            "network_id": NETWORK_ID,
            "instance_id": INSTANCE_ID,
        }
        params = params.model().state_dict()

        if not i:
            torch.save(
                OrderedDict({k: v for k, v in params.items() if k.startswith("core.")}),
                os.path.join(mdir, "params_core.pt"),
            )

        torch.save(
            OrderedDict({k: v for k, v in params.items() if not k.startswith("core.")}),
            os.path.join(mdir, f"params_{session}_{scan_idx}.pt"),
        )

    units = pd.concat(dfs, ignore_index=True)
    units.to_csv(os.path.join(mdir, "units.csv"), index=False)

    scans = pd.DataFrame(scans)
    scans.to_csv(os.path.join(mdir, "scans.csv"), index=False)

    try:
        return make_archive(mdir, "zip", mdir)
    finally:
        rmtree(mdir)
