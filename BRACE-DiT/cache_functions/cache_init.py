def cache_init(model_kwargs, num_steps):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache[-1]={}

    for j in range(28):
        cache[-1][j] = {}
    for i in range(num_steps):
        cache[i]={}
        for j in range(28):
            cache[i][j] = {}
    cache_dic['cache']                = cache
    cache_dic['flops']                = 0.0
    cache_dic['interval']             = model_kwargs['interval']
    cache_dic['max_order']            = model_kwargs['max_order']
    cache_dic['test_FLOPs']           = model_kwargs['test_FLOPs']
    cache_dic['first_enhance']        = 2
    cache_dic['cache_counter']        = 0
    cache_dic['time_steps'] = model_kwargs['time_steps']
    # 新增：保存 forward 过程中真实 diffusion timestep
    cache_dic['real_timestep_log'] = []
    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [49]
    current['period'] = 1
    current['mode']= 'taylor'
    current['scale'] = 1
    current['real_timestep'] = None
    current['activated_steps_real'] = None
    # 可以是bary，taylor，fora，full；
    return cache_dic, current

