import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from utils.data_utils import get_loader, test_custom_loader
from utils.utils import dice, resample_3d

from monai.inferers import sliding_window_inference
from layers.swin3d_layer import SwinTransformerForClassification

def load_3D_Swin_TF(best_model_pth):    

    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinTransformerForClassification(
        img_size=64,
        num_classes = 2,
        in_channels=1,
        out_channels=768, ## output of feature map channels
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0
    )

    model_dict = torch.load(best_model_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()   

    return model 

def to_binary(arr, threshold, dtype=np.bool):
    arr[arr >= threshold] = 5.
    arr[arr < threshold] = 1.
    arr[arr >= 5.] = 0.
    return arr.astype(dtype)

def predict_class(model, image):
    with torch.no_grad():
        # calculate outputs by running images through the network
        logits = model(image)
        # the class with the highest score is what we choose as prediction
        _, predicted = torch.max(logits.data, 1)
        #print(logits.data)

    return predicted

def process_file(model, file_path, category, results_list):
    
    
    #predict_df = pd.DataFrame(columns=columns)

    classes = ['TP', 'FP']

    for file in os.listdir(file_path):
        npz_path = os.path.join(file_path, file)
        print('npz_path:', npz_path)
        npz_file = np.load(npz_path)
        print(npz_file['centroid'])

        if 'patient_id' in npz_file:
            patient_id = npz_file['patient_id']
        else:
            patient_id = 'Unknown'

        if 'TP' in npz_path:
            if str(npz_file['nodule_type']) in ['HE', 'HO']:
                nodule_type = 'NS'
            else :
                nodule_type = str(npz_file['nodule_type']) 
        else :
            nodule_type = 'NA'

        # predict
        image = np.empty((1, 1, 64, 64, 64))
        image[0, 0, :] = npz_file['image']
        #image = np.expand_dims(image, axis=0).astype(np.float32)  
        image = torch.tensor(image, dtype=torch.float)            

        # predict the class (not the prediction score, so I didn't set any threshold)      
        predicted = predict_class(model, image)
        predicted_class = classes[predicted]
        #print(predicted_class)
        
        #row_information = [patient_id , npz_file['series_uid'], npz_file['contrast'], npz_file['centroid'], npz_file['nodule_volume'] if category=='FP' else 'Nan', nodule_type, npz_file['lesion_ID'] if category=='TP' else 'Nan', category, predicted_class]
        row_information = [npz_path, patient_id , npz_file['series_uid'], npz_file['contrast'], npz_file['centroid'], 'Nan', nodule_type, npz_file['lesion_ID'] if category=='TP' else 'Nan', category, predicted_class]
        results_list.append(row_information)
        #predict_df = pd.concat([predict_df, pd.DataFrame([row_information], columns = columns)], ignore_index=True)

    

def predict_all_data(test_dir, best_model_pth, export_to_csv=True, save_csv_dir='Evaluate/predict_table'):  
    model = load_3D_Swin_TF(best_model_pth)
    columns = ['Path', 'patient_id', 'series_uid', 'contrast', 'centroid', 'nodule_volume', 'nodule_type', 'lesion_ID', 'True label', 'Predict_outcome']
    results_list = []

    for category in os.listdir(test_dir):         
        if category == 'FP':
            category_path = os.path.join(test_dir, category)
            for batch in os.listdir(category_path):
                file_path = os.path.join(category_path, batch)
                process_file(model, file_path, category, results_list)
        else:
            file_path = os.path.join(test_dir, category)
            process_file(model, file_path, category, results_list)
    
    predict_df = pd.DataFrame(results_list, columns=columns)

    if export_to_csv : 
        filename = best_model_pth.split('/')[-2]
        predict_df.to_csv(os.path.join(save_csv_dir, f'{filename}.csv'), index=False)

    return predict_df

def predict_class_threshold(model, image):
    import torch.nn.functional as F
    with torch.no_grad():
        # calculate outputs by running images through the network
        logits = model(image)
        # the class with the highest score is what we choose as prediction
        #_, predicted = torch.max(logits .data, 1)
        probabilities = F.softmax(logits, dim=1)

    return probabilities


def predict_all_data_and_plot_ROC(test_dir, best_model_pth, export_to_csv=True, save_csv_dir='Evaluate/predict_table'):  
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    model = load_3D_Swin_TF(best_model_pth)
    # columns = ['patient_id', 'series_uid', 'contrast', 'centroid', 'nodule_volume', 'nodule_type', 'lesion_ID', 'True label', 'Predict_outcome']
    # predict_df = pd.DataFrame(columns=columns)

    classes = ['TP', 'FP']
    thresholds = np.linspace(0, 1, 11)

    model_name =  best_model_pth.split('/')[-2]

    true_labels= []
    probabilities = []
    for category in os.listdir(test_dir): 
        category_path = os.path.join(test_dir, category)
        for file in os.listdir(category_path):
            npz_path = os.path.join(category_path, file)
            print('npz_path:', npz_path)
            npz_file = np.load(npz_path)


            label = 0 if category == 'TP' else 1
            true_labels.append(label)

            # predict
            image = np.empty((1, 1, 64, 64, 64))
            image[0, 0, :] = npz_file['image']
            #image = np.expand_dims(image, axis=0).astype(np.float32)  
            image = torch.tensor(image, dtype=torch.float)            

            # prediction score    
            prob_score = predict_class_threshold(model, image)
            #print('Score:', prob_score)
            probabilities.append(prob_score[0][1])  # 假設第二個元素是正類的機率

    # Calculate ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    print('FP Rate: ', fpr)
    print('TP Rate', tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f'/your path/{model_name}.png')
    plt.show()
        


def plot_confusion_matrix(cm, save_dir, model_name, data_category):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FP', 'TP'])
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(save_dir, f'{model_name}_{data_category}_confusion_matrix.png'))

def Calculate_confusion_matrix(gt_ind_arr, pred_ind, data_resource, save_dir = 'Evaluate/confusion_matrix'):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(gt_ind_arr, pred_ind)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)

    print('Specificity : ', specificity)
    print('Recall : ', recall)
    print('Precision : ', precision)

    plot_confusion_matrix(cm, save_dir, model_name, data_resource)

if __name__ == '__main__':

    test_dir = "your test data path"
    best_model_pth = "your model path/model.pt"

    model_name =  best_model_pth.split('/')[-2]
    test_df = predict_all_data(test_dir, best_model_pth, export_to_csv=True, save_csv_dir='Evaluate/predict_table')
    Calculate_confusion_matrix(test_df['True label'], test_df['Predict_outcome'], data_resource='Testing')

    #predict_all_data_and_plot_ROC(test_dir, best_model_pth, export_to_csv=True, save_csv_dir='Evaluate/predict_table')