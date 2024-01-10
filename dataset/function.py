import numpy as np

def train_split(labels, n_labeled_per_class, n_unlabeled_per_class,num_classes,using_all='All'):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    print(f'len train_labeled_idxs = {len(train_labeled_idxs)} // len train_unlabeled_idxs = {len(train_unlabeled_idxs)}')
    return train_labeled_idxs, train_unlabeled_idxs

def make_imb_data(max_num, class_num, gamma,imb):
    if imb == 'long':
        mu = np.power(1/gamma, 1/(class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
    if imb=='step':
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
    return list(class_num_list)
