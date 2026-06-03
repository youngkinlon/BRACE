def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    if 'full_counter' not in cache_dic:
        cache_dic['full_counter'] = 0
    
    first_step = (current['step'] < cache_dic['first_enhance'])
        
    if (first_step) or (cache_dic['cache_counter'] == cache_dic['fresh_threshold'] - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        cache_dic['full_counter'] += 1
       
    elif (cache_dic.get('use_grouped_taylor', False)):
        current['type'] = 'grouped_taylor_cache'
        cache_dic['cache_counter'] += 1
    
    elif (cache_dic['taylor_cache']):
        current['type'] = 'taylor_cache'
        cache_dic['cache_counter'] += 1
