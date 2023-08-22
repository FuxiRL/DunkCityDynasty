# Environment Info

The state information that we receive in our game mainly consists of the following three parts:

* Infos : The information of the game, including reward event info and end values.
* Raw States : The raw state of the environment, which is in the form of a dictionary.
* Legal Action : The legal action of the player for current state.

The reward event is an event used to help users construct environmental rewards, and it is refreshed every step. And the end values are sent at the end of each round and contain information such as victory or defeat, to help users understand additional information. They will be introduced as separate parts as following like raw states.


## State

The raw state of the environment is a dictionary with the following keys:

* `global_state` : the global state of the game, including common information such as attack remain time, whether the player is home or away, etc.
* `self_state`: the state of the ball player itself, including the player's character id, position, etc.
* `ally_0_state`: the state of the first ally player $0$, similar to `self_state`.
* `ally_1_state`: the state of the second ally player $1$, similar to `self_state`.
* `enemy_0_state`: the state of the first enemy player $0$, similar to `self_state`.
* `enemy_1_state`: the state of the second enemy player $1$, similar to `self_state`.
* `enemy_2_state`: the state of the third enemy player $2$, similar to `self_state`.

The more in-depth information of `global_state` and agent states (including `self_state` and the other $5 $players) is shown in the following tables.

### Global State

| Variable Name           | Description                                                  | Range                                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| attack_remain_time      | Remaining attack time                                        | [0,20.0)                                                     |
| match_remain_time       | Remaining  match time                                        | [0,150)                                                      |
| is_home_team            | Whether the player is home team                              |                                                              |
| home_score              | Home  team score                                             | 0,2,3                                                        |
| away_score              | Away team score                                              |                                                              |
| ball_position_x(y,z)    | Three-dimensional  Cartesian coordinates of the ball.        |                                                              |
| vec_ball_basket_x(y,z)  | Cartesian  distance between the basketball hoop and the ball |                                                              |
| team_own_ball           | Whether our  team has possession of the ball                 | $1$: yes<br/>$0$: no                                         |
| enemy_team_own_ball     | Whether enemy  team has possession of the ball,              | $1$: yes<br/>$0$: no                                         |
| ball_clear              | Whether the  ball crosses the three-point line               | $1$: yes<br/>$0$: no                                         |
| ball_status             | Status of  ball                                              | $0$:None <br/>$1$:Manual<br/>$2$:Physics<br/>$3$:Shoot<br/>$4$:Owned<br/>$5$:Pass |
| can_rebound             | Whether  the ball can be rebounded                           | $1$: yes<br/>$0$: no                                         |
| dis_to_rebound_x(z,y)   | Cartesian  distance between the player and the rebound       |                                                              |
| can_block               | Whether  the ball can be blocked                             | $1$: yes<br/>$0$: no                                         |
| shoot_block_pos_x(z)    |                                                              |                                                              |
| dis_to_block_pos_x(z,y) |                                                              |                                                              |
| block_diff_angle(r)     |                                                              |                                                              |

### Agent State

| Variable Name            | Description                                                  | Range                                              |
| ------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| character_id             | player id                                                    |                                                    |
| position_type            | Position type of player                                      | $0$:C <br/>$1$:PF<br/>$2$:SF<br/>$3$:SG<br/>$4$:PG |
| buff_key                 | Dictionary Key  of Buff                                      |                                                    |
| buff_value               | Dictionary  Value of Buff                                    |                                                    |
| stature                  | Stature of  player(cm)                                       |                                                    |
| rational_shoot_distance  | Rational shoot  distance of player                           |                                                    |
| position_x(y,z)          | Cartesian  coordinates of player                             |                                                    |
| v_delta_x                | The player's  velocity on the x-axis.                        |                                                    |
| v_delta_z                | The  player's velocity on the z-axis. The basketball court plane is formed by the  x and z axes. |                                                    |
| player_to_me_dis_x(z)    |                                                              |                                                    |
| basket_to_me_dis_x(z)    |                                                              |                                                    |
| ball_to_me_dis_x(z)      |                                                              |                                                    |
| polar_to_me_angle(r)     |                                                              |                                                    |
| polar_to_basket_angle(r) |                                                              |                                                    |
| facing_x(y,z)            | Player's facing  vector.                                     |                                                    |
| block_remain_best_time   |                                                              |                                                    |
| block_remain_time        |                                                              |                                                    |
| is_out_three_line        |                                                              |                                                    |
| is_ball_owner            |                                                              |                                                    |
| own_ball_duration        |                                                              |                                                    |
| cast_duration            |                                                              |                                                    |
| power                    |                                                              |                                                    |
| is_cannot_dribble        |                                                              |                                                    |
| is_pass_receiver         |                                                              |                                                    |
| is_marking_opponent      |                                                              |                                                    |
| is_team_own_ball         |                                                              |                                                    |
| inside_defence           |                                                              |                                                    |
| player_state             |                                                              |                                                    |
| skill_state              |                                                              |                                                    |

## Reward Event

## End Values
## Action

In our environment, a total of 52 actions are reserved, of which 12 actions are common actions for all players as shown in the table below. The remainder are skill infos for each player, and the types and numbers of these skills vary, which can be referred in `Skill Info` of `Players` part.

| action index | Description         |
| ------------ | ------------------- |
| 0            | Noop                |
| 1            | Move: 90            |
| 2            | Move: 270           |
| 3            | Move: 180           |
| 4            | Move: 0             |
| 5            | Move: 135           |
| 6            | Move: 45            |
| 7            | Move: 225           |
| 8            | Move: 315           |
| 9            | Cancel Skill        |
| 10           | Pass Ball to Ally 1 |
| 11           | Pass Ball to Ally 2 |



## Players

### Jokic

#### Skill Info

| action index |      Description       |
| :----------: | :--------------------: |
|      12      |         Shoot          |
|      13      |  Post up, Pivot left   |
|      14      |  Post up, Pivot right  |
|      15      |     Call For Ball      |
|      16      |        Defense         |
|      17      |        Rebound         |
|      18      |         Block          |
|      19      |         Steal          |
|      20      |       Accelerate       |
|      21      | Running alley-oop pass |
|      22      |       Jump Ball        |
|      23      |   Dream Shake First    |
|      24      |   Dream Shake Second   |
|      25      |    High Vision Pass    |
|      26      |      Soft Floater      |
|      27      |      Heavy Screen      |
|      28      |      Stable Layup      |
|      29      |   Jokic's Post Move    |
|      30      |  Heavyweight Box Out   |
|      31      |       Slick Pass       |
|      32      |    Hook Shot(Left)     |
|      33      |    Hook Shot(Right)    |
|      34      |       Quick Shot       |

### Zion

#### Skill Info

### James

#### Skill Info

| action index |      Description       |
| :----------: | :--------------------: |
|      12      |      Drive  Left       |
|      13      |      Drive Right       |
|      14      |     Call For Ball      |
|      15      |         Screen         |
|      16      |        Defense         |
|      17      |        Rebound         |
|      18      |         Steal          |
|      19      |         Cover          |
|      20      |       Accelerate       |
|      21      |       Jump Ball        |
|      22      |      James' Shot       |
|      23      | Tank Turnaround(Left)  |
|      24      | Tank Turnaround(Right) |
|      25      |       Full Block       |
|      26      |    Focus-3-Pointer     |
|      27      |      Tank Charge       |
|      28      |     Drive Fadeaway     |
|      29      |  Post King(2Pt Left)   |
|      30      |  Post King(2Pt Right)  |
|      31      |   3Pt King(3Pt Left)   |
|      32      |  3Pt King(3Pt Right)   |
|      33      |     Tank Dunk(Far)     |
|      34      |   Turnaround Charge    |
|      35      |    One-Handed Dunk     |
|      36      |  Turnaround Fadeaway   |

### Curry

#### Skill Info
### Donic

#### Skill Info

| action index |         Description         |
| :----------: | :-------------------------: |
|      12      |            Shoot            |
|      13      |         Drive Left          |
|      14      |         Drive Right         |
|      15      |        Call For Ball        |
|      16      |           Screen            |
|      17      |           Defense           |
|      18      |           Rebound           |
|      19      |            Block            |
|      20      |            Steal            |
|      21      |            Cover            |
|      22      |         Accelerate          |
|      23      |   Running alley-oop pass    |
|      24      |          Jump Ball          |
|      25      |       Drive in place        |
|      26      |    Magic Pass(hook pass)    |
|      27      | Magic Pass(drive and dish)  |
|      28      |  Inside-out 3-Point(left)   |
|      29      |  Inside-out 3-Point(right)  |
|      30      |          Step-back          |
|      31      |      Hesitation Drive       |
|      32      |      Turnaround Finish      |
|      33      |        Post Fadeaway        |
|      34      |      Slide Step Drive       |
|      35      | Changeable 3-Pointer(left)  |
|      36      | Changeable 3-Pointer(right) |
|      37      |       Half-Turnaround       |
|      38      |       Fake Up-and-Und       |
|      39      |     Fadeaway Up-and-und     |


