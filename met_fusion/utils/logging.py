import os
import numpy as np

def make_init(log_path, num_class, top_k):
    acc = 0.
    f1_macro = 0.
    confusion = str(np.zeros([num_class, num_class]))

    result_report = '[Result]\n' \
                   'Accuracy : %.3f, F1_Macro : %.3f\n' \
                   '[Confusion]\n' \
                   '%s\n' \
                   '[Condition]\n' \
                   '\n' \
                   '\n' \
                   '\n' \
                   '---------------\n'%(acc, f1_macro, str(confusion))

    result_report = result_report * top_k
    with open(log_path, 'w', encoding='utf') as f:
        f.writelines(result_report)

def log(save_folder, condition, result, option, target, high=True, gpu_num=None):
    # Check the log Existance
    folder_name = save_folder
    os.makedirs(folder_name, exist_ok=True)

    if gpu_num is not None:
        log_path = os.path.join(folder_name, 'log%s.txt' %gpu_num)
    else:
        log_path = os.path.join(folder_name, 'log.txt')

    if os.path.isfile(log_path) is not True:
        print('Generate New log files')
        make_init(log_path, option['num_class'], option['top_k'])

    ## Report Data
    # Result
    argument = tuple(list(result.values()) + list(condition.values())) # acc, f1_macro, str(confusion), exp_num, epoch, lr, lr2, alpha, scheduler, sampler_type

    # Re-format to text lines
    result_report = '[Result]\n' \
                   'Accuracy : %.3f, F1_Macro : %.3f\n' \
                   '[Confusion]\n' \
                   '%s\n' \
                   '[Condition]\n' \
                   'epoch : %d, lr: %.5f, fusion: %s, \n' \
                   'train_rule : %s, sampler_type : %s, \n' \
                   'normalize : %s, loss : %s \n' \
                   '---------------\n' %argument

    # Comparing Results
    with open(log_path, 'r', encoding='utf') as f:
        file = f.readlines()
        file = np.asarray(file)

    res_imp = np.zeros([option['top_k']])
    top_ix = np.where(np.asarray(file) == '[Result]\n')[0]

    for k, result_ix in enumerate(top_ix):
        ix = result_ix + 1
        res = [list(map(str.strip, file_.split(':'))) for file_ in file[ix].split(',')]
        res = np.asarray(res)

        res_imp[k] = float(res[int(np.where(res == target)[0]), 1])

    if high == True:
        min_ix = np.argmin(res_imp)
        result_report = result_report.split('\n')[:-1]
        result_length = len(result_report)

        if result[target] > res_imp[min_ix]:
            result_report = [r.strip() + '\n' for r in result_report]
            file = np.delete(file, range(top_ix[min_ix],(top_ix[min_ix] + result_length)), 0)
            file = np.asarray(file.tolist() + result_report)
            save = True
        else:
            save = False
    else:
        max_ix = np.argmin(res_imp)
        result_report = result_report.split('\n')[:-1]
        result_length = len(result_report)

        if result[target] < res_imp[max_ix]:
            result_report = np.asarray([r.strip() + '\n' for r in result_report])
            file[top_ix[max_ix]:(top_ix[max_ix]+result_length)] = result_report
            save = True
        else:
            save = False

    # Write the results
    if save == True:
        file = ''.join(file)
        with open(log_path, 'w', encoding='utf') as f:
            f.writelines(file)
