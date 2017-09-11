This MATLAB package provides the MLAPG algorithm proposed in our ICCV 2015 paper [1], and a demo showing its usage and evaulation on the VIPeR database.

The package contains the following components. MLAPG.m is the MATLAB function for the MLAPG algorithm. Demo.m is a demo showing how to use the algorithm and how to evaluate its performance on the VIPeR database. Three functions for computing the Euclidean distance, PCA subspace, and the CMC curves are also provided for the evaluation purpose.

For a quick start, run the Demo.m code for metric learning and performance evaluation on the VIPeR database. Note that to run the demo you need to download the extracted LOMO features on the VIPeR database on http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/, and place it in the same folder of this code. You can run this script to reproduce our ICCV 2015 results on VIPeR.



Version: 1.0
Date: 2015-12-07
 
Author: Shengcai Liao
Institute: National Laboratory of Pattern Recognition, 	Institute of Automation, Chinese Academy of Sciences

Email: scliao@nlpr.ia.ac.cn

Project page: http://www.cbsr.ia.ac.cn/users/scliao/projects/mlapg/


References:

[1] Shengcai Liao and Stan Z. Li, "Efficient PSD Constrained Asymmetric Metric Learning for Person Re-identification." In IEEE International Conference on Computer Vision (ICCV 2015), December 11-18, Santiago, Chile, 2015.
