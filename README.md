# Deep multi-omic network fusion for marker discovery of Alzheimer’s Disease
Linhui Xie[1],†, Yash Raj[2],†, Pradeep Varathan[2], Bing He[2], Kwangsik Nho[3],
Paul Salama[1], Andrew J. Saykin[3], and Jingwen Yan[2][3],∗ <br/>
[1] Department of Electrical and Computer Engineering, Indiana University Purdue University Indianapolis, Indianapolis, IN 46204, USA, <br/>
[2] Department of BioHealth Informatics, Indiana University Purdue University Indianapolis, Indianapolis, IN 46204, USA,<br/>
[3] Department of Radiology and Imaging Sciences, Indiana University School of Medicine, Indianapolis, IN 46204, USA <br/>
∗ To whom correspondence should be addressed.<br/>
† Contributed equally.

This repository was initially forked from Varmole. {https://github.com/namtk/Varmole}

## Abstract

Multi-omic data spanning from genotype, gene expression to protein expression have been
increasingly explored, with attempt to better interpret genetic findings from genome wide association
studies and to gain more insight of the disease mechanism. However, gene expression and protein
expression are part of dynamic process changing in various ways as a cell ages. Expression data captured
by existing technology is often noisy and only capture a screenshot of the dynamic process. Performance
of models built on top of these expression data is undoubtedly compromised. To address this problem, we
propose a new interpretable deep multi-omic network fusion model (MoFNet) for predictive modeling of
Alzheimer’s disease. In particular, the information flow from DNA to protein is leveraged as a prior multi-
omic network to enhance the signal in gene and protein expression data so as to achieve better prediction
power. The proposed model MoFNet significantly outperformed all other state-of-art classifiers when
evaluated using genotype, gene expression and protein expression data from the ROS/MAP cohort.
Instead of individual markers, MoFNet yielded 3 major multi-omic subnetworks related to innate immune
system, clearance of unwanted cells or misfolded proteins, and neurotransmitter release respectively.

                 

## License
MIT License

Copyright (c) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
