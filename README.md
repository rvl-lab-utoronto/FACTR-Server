
<h1> FACTR Rizon Teleop Setup</h1>

#### Adapted from [FACTR Teleop](https://github.com/JasonJZLiu/FACTR_Teleop)

<br>

## Catalog
- [Communication Diagram](#communication-diagram)
- [Installation](#installation)
- [FACTR Teleop](#factr-teleop)


## Communication Diagram
```mermaid
graph LR
    %% Nodes definition

    subgraph B [Factr Firmware]
        A[Factr]
    end
    C([ROS2 Topics])
    subgraph E [Flexiv SDK]
        D[Rizon 4S]
    end

    %% Connections
    B <--> C
    C <--> D
    
    %% Styling (Optional)
    style C fill:#f9f,stroke:#333,stroke-width:2px
```

## Installation
If you run into errors while running the code (i.e., missing dependencies) download the missing stuff along the way. can't help ya ☝️
- Install [ROS2-Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- Download this repo to a workspace (i.e., a dir): `git clone https://github.com/rvl-lab-utoronto/force-vla.git`
- Switch to branch FactrRizonTeleop
- Follow the setup guide in [FACTR_Teleop](https://github.com/JasonJZLiu/FACTR_Teleop/README.md)
- We will not be using Dynamixel Wizard!


### ROS 2 Packages

There are four ROS 2 packages in this repository:

- `factr_teleop`
- `bc`
- `cameras`
- `python_utils`

you can find them in `/src`


### ROS 2 Workspace Setup

These packages must reside within a **ROS 2 workspace**. If you do not already have one, create a workspace by following the [ROS 2 workspace tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html).

Then:

2. Ensure to source the ROS2 setup script in your terminal
   ```bash
   source /opt/ros/humble/setup.bash # or zsh
   ```
   Note that this command should be run everytime you open a new terminal.
3. From the root of your workspace, build the workspace via:
   ```bash
   colcon build 
   ```
   This should create the following folders in your workspace root
   ```bash
   build  install  log  src
   ```
4. From the root of your workspace, source the overlay via
   ```bash
   source install/local_setup.bash # or zsh
   ```
   Note that this command should also be run everytime you open a new terminal.

> For more guidance, refer to the [ROS 2 Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html).



## FACTR Teleop

To publish just the joint positions via ROS2:

   1, Navigate to the root folder of your workspace
   
   2, run `source install/setup.bash`
   
   3, run `colcon build`
   
   4, run `ros2 run factr_teleop factr_rizon_testing`


