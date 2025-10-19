class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/public/home/lxy/bao_3/UniMod1K/SPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_models = '/public/home/lxy/bao_3/UniMod1K/SPT/pretrained_models'
        self.lasot_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/lasot'
        self.got10k_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/got10k'
        self.lasot_lmdb_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/got10k_lmdb'
        self.trackingnet_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/trackingnet'
        self.trackingnet_lmdb_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/trackingnet_lmdb'
        self.coco_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/coco'
        self.coco_lmdb_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/vid'
        self.imagenet_lmdb_dir = '/public/home/lxy/bao_3/UniMod1K/SPT/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.rgbd_dir = ''
