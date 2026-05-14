# Navigation Stack Restart Scope

## Purpose

Rebuild the TurtleBot3 Gazebo navigation stack cleanly for the FYP exploration system.

## Keep

- Gazebo world
- RTAB-Map
- OctoMap
- exploration_grid generation
- frontier detector
- exploration status monitor

## Replace / rebuild

- Nav2 config
- Nav2 goal executor
- startup script Nav2/executor entries

## Initial navigation baseline

- /selected_frontier_goal -> simple executor -> Nav2 NavigateToPose -> /cmd_vel
- Global costmap from /exploration_grid
- Local costmap from /points
- Known-free global planning only
- base_footprint robot frame
- Forward-only
- No reverse
- No path subgoals
- No FollowPath
- No goal blacklist initially
- No teleop during autonomous testing
