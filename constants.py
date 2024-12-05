DEVICE = 'cuda'

# Label mappings
LABEL_MAPPING_HAM = {
    'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
}
LABEL_MAPPING_ISIC2018 = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6
}
LABEL_MAPPING_ISIC2019 = {
    'mel': 0, 'nv': 1, 'bcc': 2, 'ak': 3, 'bkl': 4, 'df': 5, 'vasc': 6, 'scc': 7, 'unk': 8
}