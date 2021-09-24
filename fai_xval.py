import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.tensorboard import LearnerTensorboardWriter
from fastai.utils.mem import *

fastai.torch_core.defaults.device = torch.device('cuda',0)


# Set up

codes = np.array(['Other', 'Plaque', 'Lumen'], dtype=str)
path = Path('/data/jacob/vesseg/a_converted')
path_img = path/'images'
path_msk = path/'masks'
project_id = 'x_val'

# Dataframe for image list

image_df = pd.read_csv('../preprocessing/99_test_train_valid/99_train_valid_set_image_mask.csv')
image_df['image'] = image_df['image'].str.split('/').apply(lambda x: x[-1])


# batch size and image size

bs = 4
size = (512, 512)
lr = 1e-3 


# Utility functions

def get_image_mask(image_path): return path_msk/f"{Path(image_path).name.replace('__c_r', '__mask_c_r')}"

def acc(input, target): return (input.argmax(dim=1)==target.squeeze(1)).float().mean()

def new_dice(input, target, code=0):
    n = target.shape[0]
    input = (input.argmax(dim=1) == code).view(n, -1)
    targs = (target.squeeze(1) == code).view(n, -1)
    
    intersect = (input * targs).sum(dim=1).float()
    union = (input+targs).sum(dim=1).float()
    
    l = 2. * intersect / union
    l[union == 0.] = 1.
    
    return l.mean()

def dice_plaque(input, target):  return new_dice(input, target, code=1)
def dice_lumen(input, target):  return new_dice(input, target, code=2)


# Main five fold cross validation loop

for f in range(5):    
    print('Currently on fold {}'.format(str(f)))
    
    validation_filename = 'valid_file_fold_{}.csv'.format(str(f))
    
    src = (SegmentationItemList
           .from_df(image_df, path_img, 'image')
           .split_by_fname_file(validation_filename, '/home/jacob/projects/vesseg/preprocessing/99_test_train_valid/')
           .label_from_func(get_image_mask, classes=codes))
    
    data = (src
            .transform(get_transforms(flip_vert=True), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))

    # call backs
    
    csv_log = partial(CSVLogger, filename='99_x_val_fold_{}_log'.format(str(f)), append=True)
    save_model = partial(SaveModelCallback, monitor='valid_loss', mode='min', name='99_x_val_fold_{}_bestmodel'.format(str(f)))
    stop_train = partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.002, patience=10, mode='min')    

    learn = unet_learner(data, models.resnet34, metrics=[dice_plaque, dice_lumen, acc], callback_fns=[csv_log, save_model, stop_train])
    
    print('----- training head')
    learn.fit_one_cycle(100, slice(lr))
    learn.unfreeze()
    print('----- training all')
    learn.fit_one_cycle(500, slice(lr/500, lr/5))
        
    learn.save('99_x_val_fold_{}_learner'.format(str(f)))
    learn.export('99_x_val_fold_{}_model'.format(str(f)))