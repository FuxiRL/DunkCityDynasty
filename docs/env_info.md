# Environment Info

The state information that we receive in our game mainly consists of the following three parts:

* Infos : The information of the game, including reward event info and end values.
* Raw States : The raw state of the environment, which is in the form of a dictionary.
* Legal Actions : The legal action of the player for current state.

A typical state can be represented as follows:

```json
// just for example, not real values
{
    "[member_id]": [
        // infos
        {
            // reward shoot event info
            "shoot":{
                "self_id": 1,
                "event_id": 1,
                "position": 2,
                "me": 1,
                "ally": 0,
                "enemy": 0,
                "opponent": 0,
                "open_shoot": 1,
                ... // other info
            },
            // state event info
            "state_event":{
                "free_ball": 1,
            },
            // end values
            "end_values":{
                "id": 1,
                "win": 0,
                "delta": 0,
                "score": 0,
                "team_two_try": 1,
                ... // other info
            }
        },
        // raw states
        {
            "global_states":{
                "attack_remain_time": 10.0,
                "match_remain_time": 150.0,
                "is_home_team": 1,
                ... // other info
            },
            "self_state":{
                "character_id": 1,
                "position_type": 0,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            },
            "ally_0_state":{
                "character_id": 2,
                "position_type": 1,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            },
            "ally_1_state":{
                "character_id": 3,
                "position_type": 2,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            },
            "enemy_0_state":{
                "character_id": 4,
                "position_type": 3,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            },
            "enemy_1_state":{
                "character_id": 5,
                "position_type": 4,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            },
            "enemy_2_state":{
                "character_id": 6,
                "position_type": 5,
                "buff_key": 0,
                "buff_value": 0,
                "stature": 200,
                ... // other info
            }
        },
        // legal actions
        [1,1,1,0,0,...,0]
    ]
}

```
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

|      Variable Name      |                         Description                          |                            Range                             |
| :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   attack_remain_time    |                    Remaining attack time                     |                           [0,20.0)                           |
|    match_remain_time    |                    Remaining  match time                     |                           [0,150)                            |
|      is_home_team       |               Whether the player is home team                |                                                              |
|       home_score        |                       Home  team score                       |                            0,2,3                             |
|       away_score        |                       Away team score                        |                                                              |
|  ball_position_x(y,z)   |    Three-dimensional  Cartesian coordinates of the ball.     |                                                              |
| vec_ball_basket_x(y,z)  | Cartesian  distance between the basketball hoop and the ball |                                                              |
|      team_own_ball      |         Whether our  team has possession of the ball         |                     $1$: yes<br/>$0$: no                     |
|   enemy_team_own_ball   |       Whether enemy  team has possession of the ball,        |                     $1$: yes<br/>$0$: no                     |
|       ball_clear        |        Whether the  ball crosses the three-point line        |                     $1$: yes<br/>$0$: no                     |
|       ball_status       |                       Status of  ball                        | $0$:None <br/>$1$:Manual: occur at jumping ball<br/>$2$:Physics: ball free<br/>$3$:Shoot: in the way of shooting<br/>$4$:Owned: ball owned<br/>$5$:Pass: in the way of passing |
|       can_rebound       |              Whether  the ball can be rebounded              |                     $1$: yes<br/>$0$: no                     |
|  dis_to_rebound_x(z,y)  |    Cartesian  distance between the player and the rebound    |                                                              |
|        can_block        |               Whether  the ball can be blocked               |                     $1$: yes<br/>$0$: no                     |
|  shoot_block_pos_x(z)   |                                                              |                                                              |
| dis_to_block_pos_x(z,y) |                                                              |                                                              |
|   block_diff_angle(r)   |                                                              |                                                              |

### Agent State

|      Variable Name       |                         Description                          |                       Range                        |
| :----------------------: | :----------------------------------------------------------: | :------------------------------------------------: |
|       character_id       |                          player id                           |                                                    |
|      position_type       |                   Position type of player                    | $0$:C <br/>$1$:PF<br/>$2$:SF<br/>$3$:SG<br/>$4$:PG |
|         buff_key         |                   Dictionary Key  of Buff                    |                                                    |
|        buff_value        |                  Dictionary  Value of Buff                   |                                                    |
|         stature          |                    Stature of  player(cm)                    |                                                    |
| rational_shoot_distance  |              Rational shoot  distance of player              |                                                    |
|     position_x(y,z)      |               Cartesian  coordinates of player               |                                                    |
|        v_delta_x         |             The player's  velocity on the x-axis             |                                                    |
|        v_delta_z         | The  player's velocity on the z-axis. The basketball court plane is formed by the  x and z axes |                                                    |
|  player_to_me_dis_x(z)   |                                                              |                                                    |
|  basket_to_me_dis_x(z)   |                                                              |                                                    |
|   ball_to_me_dis_x(z)    |                                                              |                                                    |
|   polar_to_me_angle(r)   |                                                              |                                                    |
| polar_to_basket_angle(r) |                                                              |                                                    |
|      facing_x(y,z)       |                   Player's facing  vector                    |                                                    |
|  block_remain_best_time  |                                                              |                                                    |
|    block_remain_time     |                                                              |                                                    |
|    is_out_three_line     |                                                              |                                                    |
|      is_ball_owner       |                                                              |                                                    |
|    own_ball_duration     |                                                              |                                                    |
|      cast_duration       |                                                              |                                                    |
|          power           |                                                              |                                                    |
|    is_cannot_dribble     |                                                              |                                                    |
|     is_pass_receiver     |                                                              |                                                    |
|   is_marking_opponent    |                                                              |                                                    |
|     is_team_own_ball     |                                                              |                                                    |
|      inside_defence      |                                                              |                                                    |
|       player_state       |                        Player status                         |                        0-5                         |
|       skill_state        |             Current action type casted by player             |                        0-26                        |

`player_state`:

* `0`: None
* `1`: Default, standing or off-the-ball movement
* `2`: Hold, hold the ball
* `3`: Dribble, dribbling
* `4`: Castskill, casting skill
* `5`: Interrupt, stiff or controlled

`skill_state`:

| Index |  Description  | Index |   Description    | Index |         Description         |
| :---: | :-----------: | :---: | :--------------: | :---: | :-------------------------: |
|   0   |     None      |   9   |     Post up      |  18   |        Post up cross        |
|   1   | Common action |  10   |     Rebound      |  19   |       Collision stop        |
|   2   |     Block     |  11   |   Receive ball   |  20   |      Diving catch ball      |
|   3   |    Boxout     |  12   |      Screen      |  21   |         Rude boxout         |
|   4   | Call for ball |  13   |      Shoot       |  22   |          Chip out           |
|   5   |  Cross over   |  14   |      Steal       |  23   |          Dash dunk          |
|   6   |    Defence    |  15   |  Instant action  |  24   |         Jump block          |
|   7   |     Pass      |  16   | Steal forward cd |  25   |    Forced to stop action    |
|   8   |    Pick up    |  17   |  Quick defense   |  26   | Cut mechanism switch action |

## Reward Event

The Reward event mainly consists of two parts. One is events related to various key nodes, such as shooting and stealing, and each event's key name will be presented in the following table. The other is events related to continuous states, such as the ball not going out for a three-pointer.

### Node Events

There will be some common information for each node event, as shown in the following table:

| Feature Key |                  Description                   |
| :---------: | :--------------------------------------------: |
|   self_id   |                   Player id                    |
|  event_id   |                Event player id                 |
|  position   |         Position type of event player          |
|     me      |      Whether the event player is oneself       |
|    ally     |        Whether the event player is ally        |
|    enemy    |       Whether the event player is enemy        |
|  opponent   | Whether the event player is my opponent player |

Note that you need to distinguish between the event player and the player themselves. The event player refers to the subject of the event, such as the shooting event. However, the game will still send this shooting event information to each player, so that users can share information when conducting multi-agent related processing.

Reward event mainly includes the following points:

* `score`: scoring event, where a player shoots and successfully scores.
* `shoot`: shooting event, where a player take a shoot try but not necessarily goal in.
* `steal`: stealing event, where a player tries to steal the other.
* `block`: blocking event, where a player tries to block the other.
* `pick up`: picking up event, where a player tries to pick up the ball.
* `rebound`: rebounding event, where a player tries to rebound.
* `screen`: screening event, where a player tries to take a screen and roll.

And the unique features of each event are shown in the following table. Note that "screen" does not have its own unique features.

|  Event Key  |   Feature Key    |                         Description                          |
| :---------: | :--------------: | :----------------------------------------------------------: |
|  **score**  |      score       |                         Score value                          |
|  **shoot**  |    open_shoot    | Whether the player is in an open position or without inference |
|             |       two        |                Whether it is a two point try                 |
|             |      three       |               Whether it is a three point try                |
|             |     goal_in      |                    Whether it is in goal                     |
|             |   hit_percent    |                         Hit percent                          |
|             |      assist      |        Whether the player was involved in the assist         |
|             |    inference     |         Whether the player was involved inferencing          |
|             | inference_degree |                       Inference degree                       |
|  **steal**  |      target      |         Whether the player was the target of a steal         |
|             |   hit_percent    |                         Hit percent                          |
|             |     success      |                       Whether succeed                        |
|  **block**  |      target      |         Whether the player was the target of a block         |
|             |  expected_score  | Expected points (built-in value) of the player who got blocked, mainly determined by whether it was a two or three-point attempt and the degree of defensive interference |
|             |   hit_percent    |                         Hit percent                          |
|             |     success      |                       Whether succeed                        |
| **pickup**  |     success      |                       Whether succeed                        |
| **rebound** |     success      |                       Whether succeed                        |
| **screen**  |                  |                                                              |

### State Events

The state events are listed as following table: 

|     Feature Key     |                         Description                          |
| :-----------------: | :----------------------------------------------------------: |
| **not_ball_clear**  | The ball didn't go out for a three-pointer, usually happens during a transition of possession |
|    **free_ball**    |            The ball is not in any player's hands             |
| **attack_time_out** |                      Attack timing out                       |
|  **got_defended**   |                 The player is being defended                 |
|  **out_of_defend**  |              The player shakes off the defense               |
| **no_defend_shoot** |             The player shoots  without defending             |
|    **long_pass**    | Long pass, which will cause the player to become stiff in the game |
|    **pass_fail**    |                         Pass failed                          |

## End Values

There are two forms of game termination in our environment: one is when one side scores or the attacking side times out, which we call a **trucation**; the other is when one side wins the entire game, which we call **done**. The former usually takes around 100 steps, while the latter takes longer. Therefore, for efficient training, we recommend using trucation as the signal for reinforcement learning instead of done.

In addition, we provide some end values **for each trucation** to support more diverse training, as shown in the table below.

|    Feature Key    |                         Description                          |
| :---------------: | :----------------------------------------------------------: |
|        id         |                    Character id of player                    |
|        win        | Whether player team win or not. Note that both win could get "win=0" when the attacking side times out |
| team_score_panel  |              team score on the score dashboard               |
| enemy_score_panel |              enemy score on the score dashboard              |
|     win_panel     | When player team win or not according to the score on dashboard |
|   is_last_round   |                  Whether is the last round                   |
|       delta       |             Score delta, could be $0,\pm2,\pm3$              |
|    delta_panel    |       Score delta according to the score on dashboard        |
|      is_home      |           Whether the player belongs to home team            |
|  skill_type_cnt   |                Count of different skill used                 |
|     skill_var     |    Variance of the number of skills used, cound be $NaN$     |
|    my_tot_try     |                  Total shoot try of player                   |
|   my_dazhao_cnt   |                Count of ultimate skill casted                |
|    my_pass_cnt    |                  Total pass count of player                  |
|  my_rebound_cnt   |               Total rebounding count of player               |
|   my_screen_cnt   |                 Total screen count of player                 |
|   my_block_cnt    |                Total blocking count of player                |
|  my_blocked_cnt   |             Total being blocked count of player              |
|   my_steal_cnt    |                Total stealing count of player                |
|   my_stolen_cnt   |              Total being stolen count of player              |
|   my_pickup_cnt   |                Total pick up count of player                 |
|     my_score      |              Score of player, counld be $0,2,3$              |
|    my_two_try     |                Total 2-pt shoot try of player                |
|   my_three_try    |                Total 3-pt shoot try of player                |
|    team_score     |                        Score of team                         |
|   team_tot_try    |                   Total shoot try of team                    |
|   team_two_try    |                 Total 2-pt shoot try of team                 |
|  team_three_try   |                 Total 3-pt shoot try of team                 |
|  team_block_cnt   |                 Total blocking count of team                 |
| team_rebound_cnt  |                Total rebounding count of team                |
|  team_steal_cnt   |                 Total stealing count of team                 |
|  team_screen_cnt  |                  Total screen count of team                  |



## Action

In our environment, a total of 52 actions are reserved, of which 12 actions are common actions for all players as shown in the table below. The remainder are skill infos for each player, and the types and numbers of these skills vary, which can be referred in `Skill Info` of `Players` part.

| Index | Description | Index |     Description     |
| :---: | :---------: | :---: | :-----------------: |
|   0   |    Noop     |   6   |      Move: 45       |
|   1   |  Move: 90   |   7   |      Move: 225      |
|   2   |  Move: 270  |   8   |      Move: 315      |
|   3   |  Move: 180  |   9   |    Cancel Skill     |
|   4   |   Move: 0   |  10   | Pass Ball to Ally 1 |
|   5   |  Move: 135  |  11   | Pass Ball to Ally 2 |

## Players

### Jokic

#### Base Info

|                             Info                             |                            Radar                             |                            Avatar                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Name: $\text{ Jokic}$<br/>Character ID: $2$<br/>Position: $\text{C}$<br/>Stature:  $\text{211 CM}$<br/>Signature Skills:<br/>$\text{Dream Shake, Heavy Screen}$ | <img src="../figs/Jokic_radar.png" alt="James" style="zoom:50%;" /> | <img src="../figs/Jokic.jpg" alt="James" style="zoom:110%;" /> |

#### Skill Info

| Index |      Description       | Index |      Description       | Index |      Description       |
| :----------: | :--------------------: | :----------: | :--------------------: | :----------: | :--------------------: |
|      12      |         Shoot          |      20      |       Accelerate       |28|Stable Layup|
|      13      |  Post up, Pivot left   |      21      | Running alley-oop pass |29|Jokic's Post Move|
|      14      |  Post up, Pivot right  |      22      |       Jump Ball        |30|Heavyweight Box Out|
|      15      |     Call For Ball      |      23      |   Dream Shake First    |31|Slick Pass|
|      16      |        Defense         |      24      |   Dream Shake Second   |32|Hook Shot(Left)|
|      17      |        Rebound         |      25      |    High Vision Pass    |33|Hook Shot(Right)|
|      18      |         Block          |      26      |      Soft Floater      |34|Quick Shot|
|      19      |         Steal          |      27      |      Heavy Screen      |||

### Zion

#### Base Info

|                             Info                             |                            Radar                             | Avatar                                                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Name: $\text{ Zion}$<br/>Character ID: $9$<br/>Position: $\text{PF}$<br/>Stature:  $\text{198 CM}$<br/>Signature Skills:<br/>$\text{Soaring Dunk, Tiger Instinct}$ | <img src="../figs/Zion_radar.png" alt="James" style="zoom:50%;" /> | <img src="../figs/Zion.jpg" alt="James" style="zoom:110%;" /> |

####  Skill Info

| Index |      Description       | Index |      Description       | Index | Description |
| ------------ | :--------------------: | :----------: | :--------------------: | :--------------------: | :--------------------: |
|      12      |      Drive  Left       |  20   | Running alley-oop pass |  28   |   Tiger Instinct    |
|      13      |      Drive Right       |  21   |       Jump Ball        |  29   |  Double Pump Dunk   |
|      14      |     Call For Ball      |  22   |      Soaring Dunk      |  30   | Catch & Turn(Left)  |
|      15      |         Screen         |  23   |        Chip Out        |  31   | Catch & Turn(Right) |
|      16      |        Defense         |  24   |    Ferocious Steal     |  32   |   Alley-oop Pass    |
|      17      |         Rebound   |  25   |       Quick Dunk       | 33 | Rim Rattler |
|      18      |       Cover       |  26   |     Run onto ball      |          |          |
| 19 | Accelerate | 27 | Leapstep Block | | |

### James

#### Base Info

|                             Info                             |                            Radar                             |                            Avatar                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Name: $\text{ James}$<br/>Character ID: $1$<br/>Position: $\text{SF}$<br/>Stature:  $\text{206 CM}$<br/>Signature Skills:<br/>$\text{Post King, Two-Way Play}$ | <img src="../figs/James_radar.png" alt="James" style="zoom:50%;" /> | <img src="../figs/James.jpg" alt="James" style="zoom:110%;" /> |

#### Skill Info

| Index |      Description       | Index |      Description       | Index | Description |
| :----------: | :--------------------: | :----------: | :--------------------: | :--------------------: | ---------------------- |
|      12      |      Drive  Left       |  21   |       Jump Ball        |       30       |       Post King(2Pt Right)       |
|      13      |      Drive Right       |  22   |      James' Shot       |    31   | 3Pt King(3Pt Left) |
|      14      |     Call For Ball      |  23   | Tank Turnaround(Left)  |      32     | 3Pt King(3Pt Right) |
|      15      |         Screen         |  24   | Tank Turnaround(Right) |     33     |     Tank Dunk(Far)     |
|      16      |        Defense         |  25   |       Full Block       |  34 | Turnaround Charge |
|      17      |        Rebound         |  26   |    Focus-3-Pointer     |  35  |  One-Handed Dunk  |
|      18      |         Steal          |  27   |      Tank Charge       |   36   |   Turnaround Fadeaway   |
|      19      |         Cover          |  28   |     Drive Fadeaway     |     |     |
|      20      |       Accelerate       |  29   |  Post King(2Pt Left)   |          |          |

### Thompson

#### Base Info

|                             Info                             |                            Radar                             | Avatar                                                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Name: $\text{ Thompson}$<br/>Character ID: $82$<br/>Position: $\text{SG}$<br/>Stature:  $\text{198 CM}$<br/>Signature Skills:<br/>$\text{Stable 3pt, Rhythm Reader}$ | <img src="../figs/Thompson_radar.png" alt="James" style="zoom:50%;" /> | <img src="../figs/Thompson.jpg" alt="James" style="zoom:110%;" /> |

#### Skill Info

| Index |      Description       | Index |        Description        | Index | Description            |
| :---: | :--------------------: | :---: | :-----------------------: | :---: | ---------------------- |
|  12   |         Shoot          |  24   |        Accelerate         |  36   | Transtion Pull-up 3(2) |
|  13   |       Drive Left       |  25   | Catch & Shoot (Call Ball) |  37   | Pass & Dash            |
|  14   |      Drive Right       |  26   |       Catch & Shoot       |  38   | Drift Pass (Dunk)      |
|  15   |     Call For Ball      |  27   |        Drift Shot         |  39   | Drift Pass (Shoot)     |
|  16   |         Screen         |  28   |    Sliding Disruption     |  40   | Dash & Accelerate      |
|  17   |        Defense         |  29   |      Fake Shot Pass       |  41   | Pull-Up Drift          |
|  18   |        Rebound         |  30   |       Rhythmic Dash       |  42   | Stable 3pt(1)          |
|  19   |         Block          |  31   |         Cut Layup         |  43   | Stable 3pt(2)          |
|  20   |         Cover          |  32   |       Rhythm Reader       |  44   | Stable 3pt(3)          |
|  21   | Running alley-oop pass |  33   |   Rhythm Reader (Dunk)    |  45   | Pass & Dash(1)         |
|  22   |       Jump Ball        |  34   |        Stable 3pt         |  46   | Pass & Dash(2)         |
|  23   |     Thompson Drive     |  35   |  Transtion Pull-up 3(1)   |       |                        |


### Curry

#### Base Info

|                             Info                             |                            Radar                             | Avatar                                                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| Name: $\text{ Curry}$<br/>Character ID: $3$<br/>Position: $\text{PG}$<br/>Stature:  $\text{188 CM}$<br/>Signature Skills:<br/>$\text{Curry Gravity, Never Give Up}$ | <img src="../figs/Curry_radar.png" alt="James" style="zoom:50%;" /> | <img src="../figs/Curry.jpg" alt="Curry" style="zoom:110%;" /> |

#### Skill Info

| Index |  Description  | Index |        Description        | Index |       Description       |
| :---: | :-----------: | :---: | :-----------------------: | :---: | :---------------------: |
|  12   |  Drive  Left  |  21   | Running alley-oop pass 1  |  30   |          Dash           |
|  13   |  Drive Right  |  22   |        Accelerate         |  31   |      Curry Gravity      |
|  14   | Call For Ball |  23   | Running alley-oop pass 2  |  32   |          Shoot          |
|  15   |    Screen     |  24   |         Jump Ball         |  33   |  Reverse Running(Back)  |
|  16   |    Defense    |  25   |      Behind Dribble       |  34   |  Reverse Running(Left)  |
|  17   |    Rebound    |  26   |       Catch & Shoot       |  35   | Reverse Running(Right)  |
|  18   |     Block     |  27   | Sidestep 3-pointer(Left)  |  36   |    Turn & Pull-back     |
|  19   |     Steal     |  28   | Sidestep 3-pointer(Right) |  37   |      Turn & Shoot       |
|  20   |     Cover     |  29   |     Soft Finger Roll      |  38   | Back Dribble Hesitation |

## Built-in Rules

To facilitate quick training for users, we have built-in two rules in the game:

* `Rule passing the ball beyond the three-point line`: The action of crossing the three-point line mainly occurs during player transitions, and although this behavior model can be learned, it requires a lot of time and is not the main focus of the game.
* `Rule passing when the ball is dead`: like crossing the three-point line, is also difficult to learn but not a key focus of AI training.