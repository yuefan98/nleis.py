# nleis.py

Second-harmonic nonlinear electrochemical impedance spectroscopy (2nd-NLEIS), a special form of nonlinear electrochemical impedance spectroscopy (NLEIS), is emerging as a powerful complementary technique to traditional electrochemical impedance spectroscopy (EIS). It retains the experimental simplicity of EIS while providing additional physical insights. However, its adoption and application have been limited by the lack of open-source, user-friendly software.

`nleis.py` aims to address this gap by providing a Python toolbox that is designed to work with and extend the capabilities of `impedance.py`. Key features include:

-   Nonlinear equivalent circuit modeling (nECM)
-   Simultaneous analysis of EIS and 2nd-NLEIS data

This repository contains the most recent version of nleis.py. As of today, nleis.py supports the latest version of `impedance.py (v1.7.1)`.

### Installation

The `nleis.py` is avaliable in a standalone version now. You can install it directly with pip.

```bash
pip install nleis
```

See [Getting started with nleis.py](https://nleispy.readthedocs.io/en/latest/getting-started.html) for instructions on how to get most of this toolbox.

In the near future, you will be able to access all the funcationality for `nleis.py` from `impedance.py`.

#### Dependencies

`nleis.py` requires the same dependencies as `impedance.py` puls the latest version of `impedance.py`:

-   Python (>=3.8)
-   SciPy (>=1.0)
-   NumPy (>=1.14)
-   Matplotlib (>=3.0)
-   Altair (>=3.0)
-   impedance(>=1.7.1)
-   pandas (>= 2.0.2)


#### Examples and Documentation

The detailed documentation can be found at [nleispy.readthedocs.io](https://nleispy.readthedocs.io/en/latest).

### Contributing to nleis.py

The nleis.py project welcomes all kinds of contributions, including bug fixes, feature requests, code reviews, new features, examples, documentation improvements, and community engagement. For any changes involving the repository, please refer to the detailed guidance in the [`CONTRIBUTING.md`](https://github.com/yuefan98/nleis.py/blob/main/CONTRIBUTING.md). If you encounter any issues or have suggestions, feel free to submit an [issue](https://github.com/yuefan98/nleis.py/issues) to let us know.

We are also excited to see contributions that expand the capabilities of `nleis.py`. Potential future features include:

-   EIS and 2nd-NLEIS data processing from the time domain
-   Data validation for 2nd-NLEIS 

### Credits
----------------------------------------------------------------

This work adopted and built the `nleis.py` based on [impedance.py](https://github.com/ECSHackWeek/impedance.py) (Murbach, M., Gerwe, B., Dawson-Elli, N., & Tsui, L. (2020). impedance.py: A Python package for electrochemical impedance analysis. Journal of Open Source Software, 5. https://doi.org/10.21105/joss.02349)

----------------------------------------------------------------
### Contributors :battery:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yuefan98"><img src="https://avatars.githubusercontent.com/u/97193085?v=4?s=100" width="100px;" alt="Yuefan Ji"/><br /><sub><b>Yuefan Ji</b></sub></a><br /><a href="#design-yuefan98" title="Design">🎨</a> <a href="https://github.com/yuefan98/nleis.py/commits?author=yuefan98" title="Code">💻</a> <a href="https://github.com/yuefan98/nleis.py/commits?author=yuefan98" title="Documentation">📖</a> <a href="https://github.com/yuefan98/nleis.py/commits?author=yuefan98" title="Tests">⚠️</a> <a href="https://github.com/yuefan98/nleis.py/pulls?q=is%3Apr+reviewed-by%3Ayuefan98" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://mattmurbach.com"><img src="https://avatars.githubusercontent.com/u/9369020?v=4?s=100" width="100px;" alt="Matt Murbach"/><br /><sub><b>Matt Murbach</b></sub></a><br /><a href="https://github.com/yuefan98/nleis.py/commits?author=mdmurbach" title="Code">💻</a> <a href="https://github.com/yuefan98/nleis.py/pulls?q=is%3Apr+reviewed-by%3Amdmurbach" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dt-schwartz"><img src="https://avatars.githubusercontent.com/u/32350188?v=4?s=100" width="100px;" alt="Dan Schwartz"/><br /><sub><b>Dan Schwartz</b></sub></a><br /><a href="https://github.com/yuefan98/nleis.py/commits?author=dt-schwartz" title="Documentation">📖</a> <a href="https://github.com/yuefan98/nleis.py/pulls?q=is%3Apr+reviewed-by%3Adt-schwartz" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->