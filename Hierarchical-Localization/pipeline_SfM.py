#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, match_dense


# ## Setup
# In this notebook, we will run SfM reconstruction from scratch on a set of images. We choose the [South-Building dataset](https://openaccess.thecvf.com/content_cvpr_2013/html/Hane_Joint_3D_Scene_2013_CVPR_paper.html) - we will download it later. First, we define some paths.

# In[ ]:


images = Path('datasets\Plant_less_images\images')

outputs = Path('outputs/sfm/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'

retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)
model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

#LOFRT
#dense_conf = match_dense.confs['loftr']
#features, matches = match_dense.main(dense_conf, sfm_pairs, images, export_dir=outputs)

#Use SuperPoint keypoints as anchors:
#features_sp = extract_features.main(feature_conf, images)
#features, matches = match_dense.main(dense_conf, sfm_pairs, images,
#                          export_dir=outputs,
#                          features_ref=features_sp)

# Localization:
#loc_features, loc_matches = match_dense.main(matcher_conf, loc_pairs,
#      images, export_dir=outputs, features_ref=features, max_kps=None)
