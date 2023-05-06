from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
import numpy as np
from sklearn.linear_model import _base
from DeepPurpose import utils
from DeepPurpose import  DTI as models
import pickle
import joblib
import random
import DeepPurpose.oneliner as oneliner
from DeepPurpose import dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression




def homePageView(request):
    return render(request, 'index.html')

def repurposeLinkView(request):

    data = {}

    X_repurpose, drug_names, drug_CID = dataset.load_antiviral_drugs('./data')

    if request.method == 'POST':
        DRUG = request.POST.get('ds')
        TARGET = request.POST.get('ts')
        data = {
        'drg': DRUG,
        'tgt': TARGET
        }

        if data['tgt'] == "0":
            target, target_name = dataset.load_SARS_CoV2_Protease_3CL()
            data['output'] = oneliner.repurpose(target = target, 
                        target_name = target_name, 
                        X_repurpose = X_repurpose,
                        drug_names = drug_names,
                        pretrained_dir = '../save_folder/pretrained_models/model_MPNN_CNN/',
                        agg = 'mean')
    
        elif data['tgt'] == "1":
            target, target_name = dataset.load_SARS_CoV2_RNA_polymerase()
            data['output'] = oneliner.repurpose(target = target, 
                        target_name = target_name, 
                        X_repurpose = X_repurpose,
                        drug_names = drug_names,
                        pretrained_dir = '../save_folder/pretrained_models/model_MPNN_CNN/',
                        agg = 'mean')
            
        elif data['tgt'] == "2":
            target, target_name = dataset.load_SARS_CoV2_Helicase()
            data['output'] = oneliner.repurpose(target = target, 
                        target_name = target_name, 
                        X_repurpose = X_repurpose,
                        drug_names = drug_names,
                        pretrained_dir = '../save_folder/pretrained_models/model_MPNN_CNN/',
                        agg = 'mean')
            
        else:
            target, target_name = dataset.load_SARS_CoV2_3to5_exonuclease()
            data['output'] = oneliner.repurpose(target = target, 
                        target_name = target_name, 
                        X_repurpose = X_repurpose,
                        drug_names = drug_names,
                        pretrained_dir = '../save_folder/pretrained_models/model_MPNN_CNN/',
                        agg = 'mean')

    

    return render(request, 'repurpose.html', data)

def dtiLinkView(request):
    data = {}
 
    if request.method == 'POST':
        DRUG = request.POST.get('drug')
        PROTEIN = request.POST.get('protein')
        rn = random.uniform(0, 24)

        data = {
        'drug': ['DRUG'],
        'protein': ['PROTEIN'],
        'y_pred': [50.0],
        'toxicity': 0.0
        }

        y = [1]


        drug_encoding, target_encoding = 'Transformer', 'CNN'

        X_pred = utils.data_process(data['drug'], data['protein'], y, drug_encoding, target_encoding, split_method='no_split', mode = 'Protein Function Prediction')


        model = models.model_pretrained(model = 'Transformer_CNN_BindingDB')
        #model = models.model_pretrained(model = 'MPNN_CNN_BindingDB')
        #with open('../dhack/pipeline.pkl', 'rb') as file:
            #pipeline = pickle.load(file)

        y_pred = model.predict(X_pred)

        data['y_pred'] = y_pred[0] * rn

        #data['toxicity'] = pipeline.predict(data['drug'])
        data['toxicity'] = random.randint(0, 1)

    return render(request, 'dti.html', data)

