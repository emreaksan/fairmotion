import os
import glob
import argparse
import tqdm
import numpy as np
from fairmotion.data import amass, bvh
from human_body_prior.body_model.body_model import BodyModel


def main(args):
    bm_female = BodyModel(model_type="smplh", bm_path=args.amass_body_model_path+"/female/model.npz", num_betas=10)
    bm_male = BodyModel(model_type="smplh", bm_path=args.amass_body_model_path+"/male/model.npz", num_betas=10)
    bm_neutral =  BodyModel(model_type="smplh", bm_path=args.amass_body_model_path+"/neutral/model.npz", num_betas=10)
    
    amass_files = sorted(glob.glob(os.path.join(args.amass_dir, '**/*.npz'), recursive=True))
    
    for file_idx in tqdm.tqdm(range(len(amass_files))):
        try:
            amass_file = amass_files[file_idx]
            bm_gender = np.load(amass_file)["gender"]
            if bm_gender == 'female':
                print('female')
                motion = amass.load(file=amass_file, bm=bm_female)
            elif bm_gender == 'male':
                print('male')
                motion = amass.load(file=amass_file, bm=bm_male)
            else:
                print('neutral')
                motion = amass.load(file=amass_file, bm=bm_neutral)
            # Create sub directories if exist.
            rel_path = os.path.relpath(amass_file, args.amass_dir)
            sub_dir = os.path.join(args.bvh_dir, os.path.split(rel_path)[0])
            os.makedirs(sub_dir, exist_ok=True)
            
            bvh_name = os.path.split(rel_path)[1][:-4] + ".bvh"  # .npz to .bvh
            bvh_file = os.path.join(sub_dir, bvh_name)
            bvh.save(motion, bvh_file)
        except Exception as e:
            print(e)
            print("Source file causing an error: " + amass_file)
  
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BVH file with block body"
    )
    parser.add_argument("--amass-dir", type=str, required=True)
    parser.add_argument("--bvh-dir", type=str, required=True)
    parser.add_argument("--amass-body-model-path", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
