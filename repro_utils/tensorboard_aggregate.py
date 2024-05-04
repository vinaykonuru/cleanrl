import tyro
import os
from typing import Literal
from dataclasses import dataclass
import tensorboard_reducer as tbr
from glob import glob
from pathlib import Path



@dataclass
class Args:
    runs_subfolder: str # assume this is a subfolder of the runs folder

# Main function
def main(args):
    # Ensure output folder exists    
    runs_path = Path(args.runs_subfolder)

    # runs/{envname_eval}/{env_name} --> output folder
    output_folder = str(runs_path / "reduced")

    for choice in ['dgum', 'ddpg', 'td3','sac']:
        input_event_dirs = sorted(glob(f"{args.runs_subfolder}/*{choice}*"))
        events_dict = tbr.load_tb_events(input_event_dirs, verbose=True, strict_tags=False, strict_steps=False)
        events_out_dir = f"{output_folder}-{choice}"
        overwrite = True
        reduce_ops = ("mean", "min", "max", "std", "median")
        reduced_events = tbr.reduce_events(events_dict, reduce_ops, verbose=True)
        tbr.write_tb_events(reduced_events, events_out_dir, overwrite, verbose=True)
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)






