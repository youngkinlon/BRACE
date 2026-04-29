
def cal_type_bary(cache_dic, current):
    '''
    Determine calculation type for this step
    '''

    # 前2步强制全量计算（初始化阶段）
    first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))
    #扩散过程的最后 N 步 ，原因是后期步骤（如接近生成完成时）需要高精度细节优化
    fresh_interval = cache_dic['interval']
    # 缓存刷新间隔（如每隔5步全量计算一次）
    # 条件1：若当前是“后期增强步骤”，或缓存计数达到刷新间隔（需更新缓存）
    activate_steps=cache_dic['time_steps']
    current_step = current['step']
    #print(activate_steps)
    next_step = 49
    for step in activate_steps:  # 假设 activate_steps 已经是排序的
        if step < current_step:
            next_step = step
            break  # 找到后立即退出循环
    acti_step=(current['step'] in activate_steps)
    # if (first_steps)  or acti_step:
    #     current['type']='full'
    #     cache_dic['cache_counter'] = 0
    #     current['activated_steps'].append(current['step'])
    #     current['period'] =int(abs(next_step- current['activated_steps'][-1]))
    #print(fresh_interval)

    if (first_steps) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        print(current['activated_steps'])
        #current['activated_times'].append(current['t'])
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Bary'

def cal_type_taylor(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    # 前2步强制全量计算（初始化阶段）
    first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))
    #扩散过程的最后 N 步 ，原因是后期步骤（如接近生成完成时）需要高精度细节优化
    fresh_interval = cache_dic['interval']
    # 缓存刷新间隔（如每隔5步全量计算一次）
    # 条件1：若当前是“后期增强步骤”，或缓存计数达到刷新间隔（需更新缓存）
    activate_steps=cache_dic['time_steps']
    current_step = current['step']
    #print(activate_steps)
    next_step = 49
    for step in activate_steps:  # 假设 activate_steps 已经是排序的
        if step < current_step:
            next_step = step
            break  # 找到后立即退出循环
    acti_step=(current['step'] in activate_steps)
    if (first_steps) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        current['activated_steps_real'] = current['real_timestep']
        #print(current['activated_steps'])
        #current['activated_times'].append(current['t'])
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'
        #current['type']='full'

def cal_type_fora(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    # 前2步强制全量计算（初始化阶段）
    first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))
    #扩散过程的最后 N 步 ，原因是后期步骤（如接近生成完成时）需要高精度细节优化
    fresh_interval = cache_dic['interval']
    # 缓存刷新间隔（如每隔5步全量计算一次）
    # 条件1：若当前是“后期增强步骤”，或缓存计数达到刷新间隔（需更新缓存）
    activate_steps=cache_dic['time_steps']
    current_step = current['step']
    #print(activate_steps)
    next_step = 49
    for step in activate_steps:  # 假设 activate_steps 已经是排序的
        if step < current_step:
            next_step = step
            break  # 找到后立即退出循环
    acti_step=(current['step'] in activate_steps)
    # if (first_steps)  or acti_step:
    #     current['type']='full'
    #     cache_dic['cache_counter'] = 0
    #     current['activated_steps'].append(current['step'])
    #     current['period'] =int(abs(next_step- current['activated_steps'][-1]))
    if (first_steps) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        #print(current['activated_steps'])
        #current['activated_times'].append(current['t'])
    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Fora'
        current['activated_steps'].append(current['step'])


def cal_type(cache_dic, current):
    current_mode=current['mode']
    if current_mode=='bary':
        cal_type_bary(cache_dic, current)
    elif current_mode=='full':
        current['type']='full'
        current['activated_steps'].append(current['step'])
    elif current_mode=='taylor':
        cal_type_taylor(cache_dic, current)
    elif current_mode=='fora':
        cal_type_fora(cache_dic, current)