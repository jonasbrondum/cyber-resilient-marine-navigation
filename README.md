# Cyber-resilient multi-modal sensor fusion for autonomous shipnavigation

This repository is a selection of code used in my master thesis with the title: **Cyber-resilient multi-modal sensor fusion for autonomous shipnavigation**. I've tried to clean the code and make it as readable as possible. Hopefully, this can be useful for anyone wanting to implement their own error-state kalman filter.

## Project abstract
Ensuring the integrity of navigation sensors is paramount for safe maritime navigation. Un- fortunately, cyber-attacks targeting satellite positioning information and onboard sensors have exposed vulnerabilities, posing a significant threat to navigation safety. In near-coast and urban environments with high traffic density and static structures, the risk of collisions and grounding is elevated. Without a navigator’s sanity check of crucial information such as position, heading, and velocity, cyber attacks could lead to accidents by compromising accurate data.

This thesis addresses the challenge of enhancing the cyber-resilience of maritime surface vessel navigation systems. Various sensor systems are evaluated, and auxiliary sensing methods are proposed to create a GNSS-independent navigation system. The thesis introduces two methods—point cloud registration and template matching—for an auxiliary positioning system. Additionally, an observer is designed to leverage the existing array of sensors within a framework supported by auxiliary measurements.

The developed navigation system is tested using real-world data collected in Copen- hagen’s inner harbor. Despite the absence of GNSS measurements, the system pro- duces sufficiently accurate positioning. The resulting position estimate demonstrates the cyber-resilience of the proposed solution for GNSS-independent navigation.

## Contents

The project consisted of several methods for calculating an odometry to use in localization.
1. LiDAR odometry from point cloud registration
2. Template matching (TM) odometry using LiDAR with electronic navigational charts (ENC) or open-street map (OSM)

These odometries were along with regular data used in the error-state Kalman filter for continuous localization.

![Project pipeline](/imgs/full_pipeline.png "Project pipeline")

## Dataset

## How-to

## Copyright

