from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.unimod1k_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/dataset/TestSet_fixed'

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/got10k_lmdb'
    settings.got10k_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/lasot_lmdb'
    settings.lasot_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/lasot'
    settings.network_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/nfs'
    settings.otb_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/OTB2015'
    settings.prj_dir = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2'
    settings.result_plot_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/test/result_plots'
    settings.results_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2'
    settings.segmentation_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/test/segmentation_results'
    settings.tc128_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/trackingNet'
    settings.uav_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/UAV123'
    settings.vot_path = '/public/home/lxy/bao_3/UniMod1K/SPT_bk2/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

