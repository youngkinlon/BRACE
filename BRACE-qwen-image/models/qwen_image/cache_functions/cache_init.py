def cache_init(method, model_kwargs=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['cache_counter'] = 0

    cache[-1]['cond']={}
    cache[-1]['uncond']={}
    for j in range(60):
        cache[-1]['cond'][j] = {
            'img_attn': {},
            'txt_attn': {},
            'img_mlp': {},
            'txt_mlp': {}
        }
        cache[-1]['uncond'][j] = {
            'img_attn': {},
            'txt_attn': {},
            'img_mlp': {},
            'txt_mlp': {}
        }
        cache_index[-1][j] = {}


    cache_dic['taylor_cache'] = False
    if method == 'original':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 1
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3

    
    elif method == 'Taylor':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = model_kwargs['interval']
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = model_kwargs['max_order']
        cache_dic['first_enhance'] = model_kwargs['first_enhance']
        cache_dic['use_grouped_taylor'] = False

    elif method == 'HiCache':
        # 启用 Hermite 多项式 HiCache 预测（通过 taylor_utils 中的分发器）
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = model_kwargs['interval']
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = model_kwargs['max_order']
        cache_dic['first_enhance'] = model_kwargs['first_enhance']
        cache_dic['use_grouped_taylor'] = False
        cache_dic['hicache_scale_factor'] = model_kwargs.get('hicache_scale', 0.5)
        cache_dic['prediction_mode'] = 'hicache'

    elif method == 'GroupedTaylor':
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = model_kwargs['interval']
        cache_dic['taylor_cache'] = True
        cache_dic['use_grouped_taylor'] = True
        cache_dic['max_order'] = model_kwargs['max_order']  # 支持最高3阶
        cache_dic['first_enhance'] = model_kwargs['first_enhance']
        cache_dic['n_clusters'] = 2
        cache_dic['group_orders'] = [1, 2]  # 初始值，会被动态计算
        cache_dic['history_window'] = 4  # 4步历史窗口
        cache_dic['save_clustering_visualization'] = False  # 启用聚类可视化
        cache_dic['current_image_idx'] = 0  # 当前图像索引
        cache_dic['one_time_clustering'] = False  # 是否200张图片只进行一次聚类
        cache_dic['clustering_count'] = 0 # one_time_clusering生效时才有意义
        
        # 初始化grouped_cache
        cache_dic['grouped_cache'] = {}
        # 为每个可能的layer_key初始化grouped_cache
        for model in ['cond', 'uncond']:
            for layer in range(60):
                for block in ['img_attn', 'txt_attn', 'img_mlp', 'txt_mlp']:
                    layer_key = (model, layer, block)
                    cache_dic['grouped_cache'][layer_key] = {} 

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['num_steps'] = model_kwargs.get('num_steps', 50)
    current['model'] = 'cond'
    current['layer'] = 0
    current['block'] = 'img_attn'
    current['mode'] = 'Brace'
    return cache_dic, current
