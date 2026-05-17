import random

def genre_sampling(arr, max_many=3, exponent=0.8):
    length = len(arr)

    if length==0:
        return ["not specified"]
    elif length==1:
        if arr[0]=="":
            return ["not specified"]
        else:
            return [arr[0]]

    how_many = min(random.randint(1, max_many),len(arr))
    if length==2:
        weights = [0.6, 0.4]
    elif length==3:
        weights = [0.6, 0.3, 0.2]
    else:
        weights = [exponent**i for i in range(length)]

    selected = []

    for _ in range(min(how_many, len(arr))):
        choice = random.choices(arr, weights=weights, k=1)[0]
        idx = arr.index(choice)
        selected.append(choice)
        del arr[idx]
        del weights[idx]

    return selected

def get_custom_metadata(info, audio):
    path = info['path']
    path_list = path.split('/')
    meta_name = path_list[-1].split('.')[0] +'.txt'
    meta_base = path_list[:-2]
    meta = '/'.join(meta_base+["meta"]+[meta_name])
    with open(meta, 'r') as f:
        line = f.readline()
        _, _, bpm, genre, style = line.split("|")

    # ==========================================Genre==============
    genre_list = genre.split("+++")
    genre_list2 = genre_sampling(genre_list)

    # ==========================================Style==============
    style_list = style.split("+++")
    style_list2 = random.sample(style_list, min(random.randint(1, 15),len(style_list)))

    # ==========================================BPM==============
    bpm_to = "{} bpm".format(bpm)

    # ==========================================Final==============
    targets = genre_list2+style_list2+[bpm_to]
    targets2 = random.sample(targets, min(random.randint(2, 20),len(targets)))

    prompt = ', '.join(targets2)

    return {"prompt": prompt}
    
    
