from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.spt.config import cfg, update_config_from_file
import re


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/spt/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # Automatically find the latest checkpoint
    checkpoint_dir = os.path.join(save_dir, "checkpoints/train/spt/unimod1k")
    if os.path.isdir(checkpoint_dir):
        # Filter for checkpoint files and sort them by epoch number
        checkpoint_list = [f for f in os.listdir(checkpoint_dir) if f.startswith('SPT_ep') and f.endswith('.pth.tar')]
        if checkpoint_list:
            # Extract epoch number and find the latest one
            latest_checkpoint = max(checkpoint_list, key=lambda f: int(re.search(r'ep(\d+)', f).group(1)))
            params.checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Automatically loading latest checkpoint: {params.checkpoint}")
        else:
            raise FileNotFoundError(f"No checkpoint files found in '{checkpoint_dir}'")
    else:
        raise FileNotFoundError(f"Checkpoint directory not found: '{checkpoint_dir}'") 

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
