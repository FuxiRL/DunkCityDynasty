
## 全局状态

| 序号 |          名称          | 范围 |               说明               |
| :--: | :--------------------: | ---- | :------------------------------: |
|  1   |   attack_remain_time   |      |           进攻剩余时间           |
|  2   |   match_remain_time    |      |           匹配剩余时间           |
|  3   |       home_score       |      |             主队得分             |
|  4   |       away_score       |      |             客队得分             |
| 5-7  |  ball_position_x(y,z)  |      |             球的坐标             |
| 8-10 | vec_ball_basket_x(y,z) |      |        篮筐与球的三维距离        |
|  11  |     team_own_ball      | 0,1  | 我方持球状态，0为不持球，1为持球 |
|  12  |  enemy_team_own_ball   | 0,1  |           敌方持球状态           |
|  13  |       ball_clear       | 0,1  |          球是否出三分线          |
|  14  |      ball_status       | 0-5  |        球的状态，0-5对应         |
|  15  |      is_home_team      |      |        是否是主队               |

## 自身状态

| 序号  |          名称           | 范围 |              说明              |
| :---: | :---------------------: | :--: | :----------------------------: |
|   1   |      character_id       |      |             角色id             |
|   2   |      position_type      | 1-5  | 角色位置，1-5对应C/PF/SF/SG/PG |
|   3   |        buff_key         |      |                                |
|   4   |       buff_value        |      |                                |
|   5   |         stature         |      |          球员身高(cm)          |
|   6   | rational_shoot_distance |      |            球员射程            |
|  7-9  |     position_x(y,z)     |      |            球员坐标            |
|  10   |        v_delta_x        |      |           x轴的速度            |
|  11   |        v_delta_z        |      |  z轴的速度，x,z轴组成球场平面  |
| 12-14 |      facing_x(y,z)      |      |                                |
|  15   |       block_time        |      |                                |
|  16   |      is_ball_owner      |      |      是否为持球者，0是1否      |
|  17   |    own_ball_duration    |      |            持球时间            |
|  18   |      cast_duration      |      |                                |
|  19   |          power          |      |           大招能量条           |
|  20   |    is_cannot_dribble    | 0,1  |   二次运球，1表示不能再运球    |
|  21   |    is_pass_receiver     | 0,1  |        是否为传球接受者        |
|  22   |    is_team_own_ball     | 0,1  |          队伍是否持球          |
|  23   |     inside_defence      |      |                                |
|  24   |      player_state       |      |            球员状态            |
|  25   |       skill_state       |      |            技能状态            |
|  26   |       can_rebound       |      |            能否篮板            |
| 27-29 |  dis_to_rebound_x(y,z)  |      |          与篮板的距离          |

## 队友以及敌人状态

两个队友+三个敌人，每个状态如下：

| 序号  |          名称           | 范围 |              说明              |
| :---: | :---------------------: | :--: | :----------------------------: |
|   1   |      character_id       |      |             角色id             |
|   2   |      position_type      | 1-5  | 角色位置，1-5对应C/PF/SF/SG/PG |
|   3   |        buff_key         |      |                                |
|   4   |       buff_value        |      |                                |
|   5   |         stature         |      |          球员身高(cm)          |
|   6   | rational_shoot_distance |      |            球员射程            |
|  7-9  |     position_x(y,z)     |      |            球员坐标            |
|  10   |        v_delta_x        |      |           x轴的速度            |
|  11   |        v_delta_z        |      |  z轴的速度，x,z轴组成球场平面  |
| 12-14 |      facing_x(y,z)      |      |                                |
|  15   |       block_time        |      |                                |
|  16   |      is_ball_owner      |      |      是否为持球者，0是1否      |
|  17   |    own_ball_duration    |      |            持球时间            |
|  18   |      cast_duration      |      |                                |
|  19   |          power          |      |           大招能量条           |
|  20   |    is_cannot_dribble    | 0,1  |   二次运球，1表示不能再运球    |
|  21   |    is_pass_receiver     | 0,1  |        是否为传球接受者        |
|  22   |    is_team_own_ball     | 0,1  |          队伍是否持球          |
|  23   |     inside_defence      |      |                                |
|  24   |      player_state       |      |            球员状态            |
|  25   |       skill_state       |      |            技能状态            |
|  26   |       can_rebound       |      |            能否篮板            |