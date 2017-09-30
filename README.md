# EquiDML
EquiDML for person re-id

Code to accompany the paper:     
Equidistance Constrained Metric Learning for Person Re-identification        
Jin Wang, Zheng Wang, Chao Liang, Changxin Gao*, Nong Sang       
Pattern Recognition.      

# Usage

Run '.\demo_viper\demo.m' for testing on VIPeR dataset with LOMO. The LOMO features are provided for testing. 

# Note

Here we present a demo on VIPeR dataset with LOMO features, however, it is easy to change to impletment on other datasets with other types of features, by extracting features of each image of the corresponding dataset and change the parameters in 'demo.m'.  Results on VIPeR, CUHK01, CUHK03, Market1501 and DukeMTMC-reID, with LOMO, GOG, IDE, and Fusion features are reported in our paper. If you have any problem to reproduce them, please cantact me.

If you find the code useful, please cite:
@article{wang2018probabilistic,
  title={Equidistance Constrained Metric Learning for Person Re-identification},
  author={Wang, jin and Wang, Zheng and Liang, Chao and Gao, Changxin and Sang, Nong},
  journal={Pattern Recognition},
  volume={74},
  pages={38--51},
  year={2018},
  publisher={Elsevier}
}

If you have any questions, please contact me: cgao@hust.edu.cn
