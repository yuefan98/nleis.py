Video Tutorial for 2nd-NLEIS
============================

The animation below illustrates how a galvanostatic impedance measurement in the time domain can
be decomposed and extracted into the linear EIS response and the 2nd-NLEIS
response. It uses a nonlinear RC circuit with asymmetric
Butler-Volmer charge transfer as a compact pedagogical model.

.. raw:: html

   <figure class="nleis-video-figure">
     <video controls playsinline preload="metadata" poster="../_static/second_harmonic_nleis_preview.jpg" style="width: 100%; border: 1px solid #d8d8cf; border-radius: 8px;">
       <source src="../_static/second_harmonic_nleis.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
     <figcaption>
       If the video does not load in your browser, open the
       <a href="../_static/second_harmonic_nleis.mp4">MP4 file directly</a>.
     </figcaption>
   </figure>

The simulation applies a zero-mean sinusoidal current perturbation,
extracts the first- and second-harmonic voltage components by FFT, then
constructs the spectra using

.. math::

   Z_1 = \frac{V_1}{\Delta I}, \qquad
   Z_2 = \frac{V_2}{(\Delta I)^2}.

The asymmetry parameter is :math:`\varepsilon = 0.1`; the perturbation amplitude is
chosen from a 5 mV linear-response target to demonstrate the coexistence of
the linear and nonlinear responses.
