# Sandbox Flooding

This repository contains the source code of a demo project that simulates
flooding in a sand box.

<iframe src="https://player.vimeo.com/video/184339357" width="320" height="240" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

The major part of the repository is an OpenCL implementation of Shallow Water
Equation (SWE) using the Smoothed Particle Hydrodynamics (SPH) method.

Becuase the dependent libraries, the FLAT (Flat Libral Art Toolkit) and
"SeeYourMusic" , are not publicly available, you can **NOT** build the demo
successfully. However, I think the source code itself is still be useful to
someone who want to implement SPH in OpenCL.

## Links

* Project idea is got from "[Augmented Reality Sandbox](http://idav.ucdavis.edu/~okreylos/ResDev/SARndbox/)" by Oliver Kreylos.
* The underlying theory (SPH + SWE) is described in these 2 papers,
  * Hyokwang Lee, Soonhung Han: [Solving the Shallow Water equations using 2D SPH particles for interactive applications](http://link.springer.com/article/10.1007%2Fs00371-010-0439-9). The Visual Computer 26, 6-8 (2010), 865–872.
  * B. Solenthaler, P. Bucher, N. Chentanez, M. Müller, M. Gross: [SPH Based Shallow Water Simulation](https://cgl.ethz.ch/publications/papers/paperSol11b.php). Proceedings of Virtual Reality Interactions and Physical Simulations (VRIPhys) (Lyon, France, December 5-6, 2011), pp. 39-46.
* OpenCL radix sorting algorithm is from "[A portable implementation of the radix sort algorithm in OpenCL](https://hal.archives-ouvertes.fr/docs/00/59/67/30/PDF/ocl-radix-sort.pdf)" by Philippe Helluy.

## License

MIT
