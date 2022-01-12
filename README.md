This is prototype code for the FASER alignment implementation in Calypso.

We align individual stations, which totals to the following alignment parameters:

* global movement of the station is not corrected here
* shift + rotation of a complete layer --> 3 sets of x, y, z, alpha, beta, gamna
* shift + rotation of inidividual modules on a layer --> 3*8 sets of x, y, z, alpha, beta, gamna

This totals to 18 + 144 alignment parameters.
