
## Global State

| No. |          Variable Name          | Range | Description |
| :--: | :--------------------: | ---- | :------------------------------: |
|  1   |   attack_remain_time   |      |           Remaining attack time           |
|  2   |   match_remain_time    |      |           Remaining match time.           |
|  3   |       home_score       |      |             Home team score             |
|  4   |       away_score       |      |             Away team score             |
| 5-7  |  ball_position_x(y,z)  |      |             Three-dimensional Cartesian coordinates of the ball.             |
| 8-10 | vec_ball_basket_x(y,z) |      |        Cartesian distance between the basketball hoop and the ball.        |
|  11  |     team_own_ball      | 0,1  | Whether our team has possession of the ball, 1 for yes, 0 for no. |
|  12  |  enemy_team_own_ball   | 0,1  |           Whether enemy team has possession of the ball, 1 for yes, 0 for no.           |
|  13  |       ball_clear       | 0,1  |          Whether the ball crosses the three-point line          |
|  14  |      ball_status       | 0-5  |        Status of ball        |
|  15  |      is_home_team      |      |      Home team or not      |

## Self State

|  No.  |      Variable Name      | Range |                         Description                          |
| :---: | :---------------------: | :---: | :----------------------------------------------------------: |
|   1   |      character_id       |       |                          Player ID                           |
|   2   |      position_type      |  1-5  |            Position type of player, C/PF/SF/SG/PG            |
|   3   |        buff_key         |       |                    Dictionary Key of Buff                    |
|   4   |       buff_value        |       |                   Dictionary Value of Buff                   |
|   5   |         stature         |       |                    Stature of player(cm)                     |
|   6   | rational_shoot_distance |       |              Rational shoot distance of player               |
|  7-9  |     position_x(y,z)     |       |               Cartesian coordinates of player                |
|  10   |        v_delta_x        |       |             The player's velocity on the x-axis.             |
|  11   |        v_delta_z        |       | The player's velocity on the z-axis. The basketball court plane is formed by the x and z axes. |
| 12-14 |      facing_x(y,z)      |       |                   Player's facing vector.                    |
|  15   |       block_time        |       |                          Block time                          |
|  16   |      is_ball_owner      |       |                  Whether is the ball owner                   |
|  17   |    own_ball_duration    |       |                       Possession time.                       |
|  18   |      cast_duration      |       |                       Skill cast time.                       |
|  19   |          power          |       |                    Ultimate skill power.                     |
|  20   |    is_cannot_dribble    |  0,1  | Double dribble, 1 indicates that dribbling is not allowed again. |
|  21   |    is_pass_receiver     |  0,1  |                 Whether is the pass receiver                 |
|  22   |    is_team_own_ball     |  0,1  |                Whether our team owns the ball                |
|  23   |     inside_defence      |       |          Whether is the tallest C/PF on the court?           |
|  24   |      player_state       |       |                       Player situation                       |
|  25   |       skill_state       |       |                       Skill situation                        |
|  26   |       can_rebound       |       |                     Whether can rebound                      |
| 27-29 |  dis_to_rebound_x(y,z)  |       |        Cartesian distance between player and rebound.        |

## States of Two Allies and Three enemies

|  No.  |      Variable Name      | Range |                         Description                          |
| :---: | :---------------------: | :---: | :----------------------------------------------------------: |
|   1   |      character_id       |       |                          Player ID                           |
|   2   |      position_type      |  1-5  |            Position type of player, C/PF/SF/SG/PG            |
|   3   |        buff_key         |       |                    Dictionary Key of Buff                    |
|   4   |       buff_value        |       |                   Dictionary Value of Buff                   |
|   5   |         stature         |       |                    Stature of player(cm)                     |
|   6   | rational_shoot_distance |       |              Rational shoot distance of player               |
|  7-9  |     position_x(y,z)     |       |               Cartesian coordinates of player                |
|  10   |        v_delta_x        |       |             The player's velocity on the x-axis.             |
|  11   |        v_delta_z        |       | The player's velocity on the z-axis. The basketball court plane is formed by the x and z axes. |
| 12-14 |      facing_x(y,z)      |       |                   Player's facing vector.                    |
|  15   |       block_time        |       |                          Block time                          |
|  16   |      is_ball_owner      |       |                  Whether is the ball owner                   |
|  17   |    own_ball_duration    |       |                       Possession time.                       |
|  18   |      cast_duration      |       |                       Skill cast time.                       |
|  19   |          power          |       |                    Ultimate skill power.                     |
|  20   |    is_cannot_dribble    |  0,1  | Double dribble, 1 indicates that dribbling is not allowed again. |
|  21   |    is_pass_receiver     |  0,1  |                 Whether is the pass receiver                 |
|  22   |    is_team_own_ball     |  0,1  |                Whether our team owns the ball                |
|  23   |     inside_defence      |       |          Whether is the tallest C/PF on the court?           |
|  24   |      player_state       |       |                       Player situation                       |
|  25   |       skill_state       |       |                       Skill situation                        |
|  26   |       can_rebound       |       |                     Whether can rebound                      |