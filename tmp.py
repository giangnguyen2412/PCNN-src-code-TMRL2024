import csv


def get_class_names(id_list):
    class_names = []
    with open('/home/lab/datasets/nabirds/classes.txt', 'r') as file:
        for line in file:
            line = line.strip()
            id = line.split(' ')[0]
            if int(id) < 1000:
                id = '0' + id
            if id in id_list:
                class_name = line.split(' ', 1)[1]
                class_names.append(class_name)
    class_names.sort()
    return class_names


overlapping_exact_match_classes = ['0879', '0911', '0828', '0884', '0853', '0895', '0533', '0875', '0897', '0865',
                                   '0802', '0939', '0900', '0864', '0868', '0910', '0831', '0896', '0947', '0449',
                                   '0936', '0887', '0807', '0923', '0808', '0810', '0870', '0876', '0859', '0903',
                                   '0858', '0479', '0915', '0450', '0803', '0862', '0827', '0866', '0830', '0474',
                                   '0476', '0902', '0941', '0534', '0886', '0824', '0917', '0898', '0549', '0877',
                                   '0448', '0885', '0944', '0863', '0946', '0957', '0832', '0860', '0871', '0867',
                                   '0872', '0559', '0950', '0925', '0913', '0851', '0834', '0937', '0954', '0471',
                                   '0889', '0553', '0932', '0561', '0804', '0835', '0857', '0861', '0470', '0800']

SCE_classes = ['0319', '0625', '0329', '0314', '0326', '0638', '0621', '0357', '0324', '0320', '0365', '0626',
               '0632', '0629', '0333', '0656', '0613', '0628', '0618', '0330', '0339', '0295', '0657', '0622',
               '0360', '0664', '0338', '0660', '0630', '0623', '0327', '0619', '0665', '0334', '0658', '0321',
               '0359', '0366', '0696', '0331', '0637', '0316', '0633', '0361', '0615', '0617', '0620', '0322',
               '0323', '0659', '0463', '0318', '0358']

SCS_classes = ['0893', '0997', '0927', '0970', '0919', '0602', '0990', '0609', '0790', '0349', '0975', '0926', '0981',
               '0345', '0644', '0940', '0772', '1000', '0348', '0931', '0446', '0755', '0942', '0789', '0949', '0768',
               '0402', '1010', '0472', '0918', '0996', '0475', '0746', '0774', '0943', '0648', '0952', '0905', '0962',
               '0793', '0447', '0979', '0395', '0948', '0888', '0892', '0647', '0783', '0799', '0945', '0763', '0891',
               '0953']

# Retrieve class names for each list
overlapping_exact_match_classes_names = get_class_names(overlapping_exact_match_classes)
SCE_classes_names = get_class_names(SCE_classes)
SCS_classes_names = get_class_names(SCS_classes)

# Create the spreadsheet
rows = zip(overlapping_exact_match_classes_names, SCE_classes_names, SCS_classes_names)

with open('class_names_spreadsheet.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['overlapping_exact_match_classes', 'SCE_classes', 'SCS_classes'])
    writer.writerows(rows)
