import torch
from assets.models import irse, ir152, facenet

def get_model_list(target_model):
    if target_model == 0:
            model_list = ['facenet','mobile_face','irse50','ir152']
    elif target_model == 1:
        model_list = ['facenet','mobile_face','ir152','irse50']
    elif target_model ==2:
        model_list = ['facenet','ir152','irse50','mobile_face']
    else:
        model_list = ['ir152','irse50','mobile_face','facenet']

    models = {}
    for model in model_list:
        if model == 'ir152':
            models[model] = []
            models[model].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('./assets/models/ir152.pth'))
            fr_model.to("cuda")
            fr_model.eval()
            models[model].append(fr_model)
        if model == 'irse50':
            models[model] = []
            models[model].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load('./assets/models/irse50.pth'))
            fr_model.to("cuda")
            fr_model.eval()
            models[model].append(fr_model)
        if model == 'facenet':
            models[model] = []
            models[model].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device="cuda")
            fr_model.load_state_dict(torch.load('./assets/models/facenet.pth'))
            fr_model.to("cuda")
            fr_model.eval()
            models[model].append(fr_model)
        if model == 'mobile_face':
            models[model] = []
            models[model].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('./assets/models/mobile_face.pth'))
            fr_model.to("cuda")
            fr_model.eval()
            models[model].append(fr_model)
    return models