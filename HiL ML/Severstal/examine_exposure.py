'''
This script serves to examine the exposure of images
It uses a json file of modeling stats (created during Validation Phase) in order to provide an overview how many flagged and unflagged images show signs of overexposure, underexposure or strong gradient.
'''

import json
import cv2
import numpy as np



def load_gray01(path):
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def metrics(arr):
    flat = arr.ravel()
    mu = float(flat.mean())
    p1, p5, p95, p99 = np.percentile(flat, [1, 5, 95, 99]).astype(np.float32)
    delta = float(p99 - p1)
    f_dark = float((flat <= (5/255)).mean())
    f_bright = float((flat >= (250/255)).mean())
    # 256-Bin Entropie
    hist = np.histogram(flat, bins=256, range=(0.0,1.0), density=True)[0]
    hist = hist + 1e-12
    H = float(-(hist*np.log(hist)).sum() * (1/256.0))  # grobe Entropie
    # Exposedness
    sigma = 0.2
    E = float(np.exp(-((flat-0.5)**2)/(2*sigma*sigma)).mean())
    return {
        "mean": mu, "p1": float(p1), "p5": float(p5),
        "p95": float(p95), "p99": float(p99),
        "dr_1_99": delta, "frac_dark": f_dark, "frac_bright": f_bright,
        "entropy": H, "exposedness": E
    }




def label_image(m, th):
    votes_under = 0
    votes_over = 0
    if m["mean"] < th["mean_low"]: votes_under += 1
    if m["mean"] > th["mean_high"]: votes_over += 1
    if m["dr_1_99"] < th["dr_min"]: votes_under += 1; votes_over += 1  # flach = beides möglich
    if m["frac_dark"] > th["frac_dark_hi"]: votes_under += 1
    if m["frac_bright"] > th["frac_bright_hi"]: votes_over += 1
    if m["exposedness"] < th["exp_min"]: votes_under += 1; votes_over += 1

    if votes_under >= 2 and votes_under > votes_over:
        return "under"
    if votes_over >= 2 and votes_over > votes_under:
        return "over"
    return "ok"


def compute_thresholds(metric_list):
    mus = np.array([m["mean"] for m in metric_list])
    drs = np.array([m["dr_1_99"] for m in metric_list])
    fd  = np.array([m["frac_dark"] for m in metric_list])
    fb  = np.array([m["frac_bright"] for m in metric_list])
    exp = np.array([m["exposedness"] for m in metric_list])

    th = {
        "mean_low": float(np.quantile(mus, 0.1)),     
        "mean_high": float(np.quantile(mus, 0.90)),    
        "dr_min": float(np.quantile(drs, 0.10)),       
        "frac_dark_hi": float(np.quantile(fd, 0.90)),  
        "frac_bright_hi": float(np.quantile(fb, 0.90)),
        "exp_min": float(np.quantile(exp, 0.10)),
    }

    # Optional: gegen sinnlose Werte mit heuristischen Mindestabständen absichern
    th["mean_low"] = min(th["mean_low"], 0.42)  # nicht zu hoch
    th["mean_high"] = max(th["mean_high"], 0.58)
    th["dr_min"] = min(th["dr_min"], 0.40)
    th["frac_dark_hi"] = max(th["frac_dark_hi"], 0.08)
    th["frac_bright_hi"] = max(th["frac_bright_hi"], 0.08)
    th["exp_min"] = min(th["exp_min"], 0.70)
    return th



def scan_files(paths: list):
    metrics_per_img = []
    for p in paths:
        im = load_gray01(p)
        m = metrics(im)
        m["path"] = p
        metrics_per_img.append(m)
    th = compute_thresholds(metrics_per_img)
    rows = []
    for m in metrics_per_img:
        lbl = label_image(m, th)
        rows.append({**m, "label": lbl})
    return rows, th




def examine_gradient(im_path: str, 
                     threshold: int = 80, 
                    horizontal_stripes: int = 10, # number of stripes along the horizontal
                    vertical_stripes: int = 10 # number of stripes along the vertical
    ):
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # partition in vertical stripes
    stripe_width = w // horizontal_stripes
    brightnesses_h = [] # brightnesses along the horizontal axis (left to right)
    for i in range(horizontal_stripes):
        start = i * stripe_width
        end = start + stripe_width
        stripe = img[:, start:end]
        brightnesses_h.append(np.mean(stripe))
    # Differenz zwischen hellstem und dunkelstem Bereich
    horizontal_range = abs(brightnesses_h[0] - brightnesses_h[-1])
    # Standardabweichung der Helligkeiten
    horizontal_std = np.std(brightnesses_h)

    # examine vertically (horizontal stripes, top to bottom)
    stripe_height = h // vertical_stripes
    brightnesses_v = [] # brightnesses along the horizontal axis (left to right)
    for i in range(vertical_stripes):
        start = i * stripe_height
        end = start + stripe_height
        stripe = img[start:end, :]
        brightnesses_v.append(np.mean(stripe))
    # Differenz zwischen hellstem und dunkelstem Bereich
    vertical_range = abs(brightnesses_v[0] - brightnesses_v[-1])
    # Standardabweichung der Helligkeiten
    vertical_std = np.std(brightnesses_v)
    
    return {
        'horizontal_graident': horizontal_range > threshold,
        'horizontal_diff': horizontal_range,
        'horizontal_std': horizontal_std,
        #'horizontal_brightnesses': brightnesses_h,
        'horizontal_min': np.argmin(brightnesses_h), 
        'horizontal_max': np.argmax(brightnesses_h) ,
        'vertical_graident': vertical_range > threshold,
        'vertical_diff': vertical_range,
        'vertical_std': vertical_std,
        #'vertical_brightnesses': brightnesses_v,
        'vertical_min': np.argmin(brightnesses_v), 
        'vertical_max': np.argmax(brightnesses_v)       
    }



dirs = ['severstal-DatasetNinja/train/img']
im_files = []
while len(dirs) > 0:
    d = dirs[0]
    for f in os.listdir(d):
        name = f'{d}/{f}'
        if os.path.isdir(name):
            dirs.append(name)
        elif f.endswith('.jpg'):
            im_files.append(name)
    dirs.remove(d)


rows, th = scan_files(im_files)    



fp = 'model_validation_unbalanced.json'
with open(fp, 'r', encoding = 'utf-8') as in_file:
    stats = json.load(in_file)


corrects = {}
wrongs = {}
keys = list(stats.keys())
for row in rows: 
    fp = row['path']
    fp = fp.split('/')[-1]
    fp = f'./data/train/img/{fp}'
    if fp in keys:
        stat = stats[fp]
        label = stat['label']
        found = False
        if stat['deviation'] == '1':
            wrongs[fp] = row
        elif stat['eff_uncertain'] == '1':
            wrongs[fp] = row
        elif stat['rn_uncertain'] == '1':
            wrongs[fp] = row
        elif (stat['eff_correct'] == '0' or stat['rn_correct'] == '0'):
            wrongs[fp] = row
        else:
            corrects[fp] = row



wrong_ok = 0
wrong_nok = 0
for w in wrongs:
    elem = wrongs[w]
    if elem['label'] == 'ok':
        wrong_ok += 1
    else:
        wrong_nok += 1


print(f'found {wrong_ok} good ims and {wrong_nok} bad ims in ims to examine')



correct_ok = 0
correct_nok = 0
for w in corrects:
    elem = corrects[w]
    if elem['label'] == 'ok':
        correct_ok += 1
    else:
        correct_nok += 1


print(f'found {correct_ok} good ims and {correct_nok} bad ims in ims to examine')



