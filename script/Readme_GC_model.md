# GC Evolution

---

This project simulate GC reactions under various conditions. We explored the effect of the following factors:

- Tether strength (force, stiffness)
- Evolution direction (xb, Gb mutability)
- Dynamic landscape (Ab feedback, dynamic force)
- Heterogeneity (diversity of binding quality)



The package consists two main parts:

- **scan_parameter**

- > A few scripts are used to scan parameters and record simulation results

- **GC evolution model**

- > Modules are used to simulate GC evolution under different conditions





## Modules

### Scan parameter

- *controller.py* 

- > Implement the **Controller** class, which provides basic functions to control a simulator. The main atrribute is **dataset**, which contains all the simulation results. It also makes it easier to save the results.

- *data.py*

- > **MyData** class provides methods to add, change and save the data. 

- *scan.py*

- > The **Scanner** class inherits properties of **Controller**. It manages GC simulations under different parameter setting. 



### GC evolution model

- model

- > GC simulation model

  - *bonds.py*

  - *singleBond.py*

  - *theory.py*

  - > The above three packages provide Brownian simulations to get extraction chance. 

  - *evolution.py*

  - >Bacic GC evolution model

  - *evolution_dynamic_force.py*

  - > GC evolution under dynamic force

  - *evolution_track_clone.py*

  - > Allows force to be heterogeneous. In addition, we can keep track different B cell clones using the package. 



