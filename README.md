# [NeurIPS 2024] Gradient-free Decoder Inversion in Latent Diffusion Models by Hong et al.
Official Repo of "Gradient-free Decoder Inversion in Latent Diffusion Models" by Hong et al.

<center>
[<a href="http://arxiv.org/abs/2409.18442">arXiv</a>] [<a href="https://smhongok.github.io/dec-inv.html">Project</a>] [<a href="https://recorder-v3.slideslive.com/#/share?share=94209&s=10070c25-7055-403c-bd99-5bfc52ab749d">video</a>] [<a href="#bibtex">bibTeX</a>]

Authors: <a href="https://smhongok.github.io/">Seongmin Hong</a><sup>1</sup>, <a href="https://www.linkedin.com/in/suhyoonjeon">Suh Yoon Jeon</a><sup>1</sup>, <a href="https://www.linkedin.com/in/khlee0192">Kyeonghyun Lee</a><sup>1</sup>, <a href="https://ernestryu.com/">Ernest K. Ryu</a><sup>2</sup>, <a href="https://icl.snu.ac.kr/pi">Se Young Chun</a><sup>1,3</sup>

<sup>1</sup>Dept. of Electrical and Computer Engineering, <sup>3</sup>INMC & IPAI, Seoul National University     
<sup>2</sup>Dept. of Mathematics, University of California, Los Angeles   
</center>

Code of a manuscript 'Gradient-free Decoder Inversion in Latent Diffusion Models'

### Environment
We used conda to run the code. We recommend running the code with the same versions of our libraries; please check `environment.yml`.

### Methods
Our main contributions (i.e., decoder inversion algorithms) are in `src/stable_diffusion/inverse_stable_diffusion.py`.

We provide two codes of experiment: 

Exp. A: Decoder inversion in SD2.1 (Figure 3)

Exp. B: Tree-rings watermarks classification (Table 2).

### Exp. A: Decoder inversion in SD2.1 (Figure 3)
To reproduce the results of Figure 3, please run
```
bash scripts/run_reconstruction.sh
```

### Exp. B: Tree-rings watermarks classification (Table 2).
To reproduce the results of Table 2, please run
```
bash scripts/run_detection.sh
```

This code is heavily based on https://github.com/YuxinWenRick/tree-ring-watermark and https://github.com/smhongok/inv-dpm.

<a name="bibtex">

### <center>BibTeX</center>
<pre> 
@misc{hong2024gradient,
      title={Gradient-free Decoder Inversion in Latent Diffusion Models}, 
      author={Seongmin Hong and Suh Yoon Jeon and Kyeonghyun Lee and Ernest K. Ryu and Se Young Chun},
      year={2024},
      eprint={2409.18442},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.18442}, 
}
</pre>
