# Activation Steering for Context-Balanced Synthetic Data

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Kirill Morozov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Sergey Karpukhin </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Aibek Alanov </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Training models on data with group imbalance, for example, “class–context” combinations, leads to a drop in quality on rare subgroups and a decrease in the fairness of predictions. One way to combat this is to supplement the training dataset with synthetic data. However, generating data via Text-to-Image (T2I) models using only a text prompt does not provide reliable control over the context and attributes of the object. In this work, we propose a framework for controlled synthetic data generation to balance groups using activation steering in Diffusion Transformer (DiT). By analogy with SHIFT, we construct steering vectors in the DiT activation space that correspond to the contexts of the original dataset, and we use them to generate examples of rare “class–context” combinations. In experiments on the NICO++ dataset, we train a CNN classifier on the original train set, balance the subsample using synthetic data while maintaining the overall size of the training set, and compare average accuracy and worst-group accuracy (WGA) with the baseline. We test the hypothesis that such controlled generation will improve WGA and fairness metrics without sacrificing average quality. The results obtained demonstrate that the proposed method can be used to enrich training data with synthetic data derived from T2I models, with the aim of improving the fairness of trained classifiers.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
