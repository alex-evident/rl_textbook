from tensor import Tensor, TensorShape
from utils import Index
from python import Python
from random.random import rand, randint
from collections.inline_list import InlineList
import time

fn build_value_dict() -> Dict[String, Int8]:
    var value_dict = Dict[String, Int8]()
    value_dict[" "] = 0
    value_dict["*"] = 1
    value_dict["S"] = 2 
    value_dict["F"] = 3 
    value_dict["C"] = 4
    return value_dict

fn read_track(filepath: String, value_dict: Dict[String, Int8]) raises -> Tensor[DType.int8]:
    with open(filepath, "r") as f:
        var grid_text = f.read()
        var grid_split = grid_text.splitlines()
        var len_y = len(grid_split)
        var len_x = len(grid_split[0])

        var track = Tensor[DType.int8].rand(TensorShape(len_y, len_x))
        for i_y in range(len_y):
            for i_x in range(len_x):
                var str_value = grid_split[len_y - i_y - 1][i_x]
                track[Index(i_y, i_x)] = value_dict[str_value]
        
        var finish_x: Int8 = -1
        var finish_y_max: Int8 = -10
        var finish_y_min: Int8 = 120
        for i_y in range(len_y):
            var row_str = grid_split[i_y].find('F')
            if row_str != -1:
                finish_x = row_str
                finish_y_max = max(finish_y_max, len_y - i_y - 1)
                finish_y_min = min(finish_y_min, len_y - i_y - 1)
        
        return track

fn get_finish_coords(filepath: String) raises -> Tuple[Int8, Int8, Int8]:
    with open(filepath, "r") as f:
        var grid_text = f.read()
        var grid_split = grid_text.splitlines()
        var len_y = len(grid_split)

        var finish_x: Int8 = -1
        var finish_y_max: Int8 = -10
        var finish_y_min: Int8 = 120
        for i_y in range(len_y):
            var row_str = grid_split[i_y].find('F')
            if row_str != -1:
                finish_x = row_str
                finish_y_max = max(finish_y_max, len_y - i_y - 1)
                finish_y_min = min(finish_y_min, len_y - i_y - 1)
        
        return finish_x, finish_y_min, finish_y_max

fn get_start_coords(filepath: String) raises -> Tuple[Int8, Int8]:
    with open(filepath, "r") as f:
        var grid_text = f.read()
        var grid_split = grid_text.splitlines()
        var len_y = len(grid_split)

        var start_x_min: Int8 = 0
        var start_x_max: Int8 = 0
        for i_y in range(len_y):
            var row_str = grid_split[i_y].find('S')
            if row_str != -1:
                start_x_min = grid_split[i_y].find('S')
                start_x_max = grid_split[i_y].rfind('S')
        
        return start_x_min, start_x_max


fn reset_state(start_coords: Tuple[Int8, Int8], inout state: SIMD[DType.int8, 4]) -> None:
    var p1 = DTypePointer[DType.int8].alloc(1)
    randint[DType.int8](p1, size=1, low=int(start_coords[0]), high=int(start_coords[1]))
    state[0] = 1
    state[1] = p1[0]
    state[2] = 0
    state[3] = 0

fn clip(value: Int, a_min: Int, a_max: Int) -> Int:
    if value < a_min:
        return a_min
    elif value > a_max:
        return a_max
    else:
        return value

fn update_state(
    inout state: SIMD[DType.int8, 4],
    inout terminated: Bool, 
    y_max: Int, 
    x_max: Int, 
    track: Tensor[DType.int8], 
    value_dict: Dict[String, Int8],
    start_coords: Tuple[Int8, Int8],
    finish_bounds: Tuple[Int8, Int8, Int8]) raises -> None:
    state[0] = clip(int(state[0] + state[2]), a_min=0, a_max=y_max)
    state[1] = clip(int(state[1] + state[3]), a_min=0, a_max=x_max)

    var new_loc = track[Index(state[0], state[1])]

    # check if finished
    if new_loc == value_dict["F"]:
        terminated = True
        return

    # check if gone out of bounds
    if new_loc == value_dict["*"]:
        # If we intersected with the finish line, we're done.
        # finish_bounds: (finish_x, finish_y_min, finish_y_max)
        var passed_finish_x = state[1] >= finish_bounds[0]
        var passed_finish_y = state[0] >= finish_bounds[1] and state[0] <= finish_bounds[2]
        if passed_finish_x and passed_finish_y:
            terminated = True
        else:
            # return the car to the start line and remove it's velocity
            reset_state(start_coords, state)

fn action_to_dv(a: Int8) -> Tuple[Int8, Int8]:
    var dv_y = a // 3 - 1
    var dv_x = a % 3 - 1
    return dv_y, dv_x

fn simulate_episode[max_steps: Int = 500](
    policy: Tensor[DType.int8], 
    inout state: SIMD[DType.int8, 4],
    inout states_hist: InlineList[SIMD[DType.int8, 4], max_steps],
    inout actions_hist: InlineList[Int8, max_steps],
    inout t: Int,
    y_max: Int, 
    x_max: Int, 
    track: Tensor[DType.int8], 
    value_dict: Dict[String, Int8],
    start_coords: Tuple[Int8, Int8],
    finish_bounds: Tuple[Int8, Int8, Int8],
    noise_prob: Float16 = 0.1, 
    eps: Float16 = 0.05,
) raises -> Bool:
    # pointers will contain the random vars - 0 indexes the epsilon roll, 1 the noise roll
    var rand_floats = DTypePointer[DType.float16].alloc(2)
    var rand_actions = DTypePointer[DType.int8].alloc(1)
    var terminated: Bool = False
    for _ in range(max_steps):
        t += 1
        # select an action
        rand(rand_floats, 2)
        var a: Int8 = 0
        if rand_floats[0] < eps: # epsilon-greedy
            rand(rand_actions, 1)
            a = rand_actions[0]
        else:
            a = policy[Index(state[0], state[1], state[2], state[3])]
        
        # save the episode info
        actions_hist.append(a)
        states_hist.append(state)

        # apply noise and select action
        var dv: Tuple[Int8, Int8] = (Int8(0), Int8(0))
        if rand_floats[1] > noise_prob:
            dv = action_to_dv(a)
        
        # apply the action to the velocity components
        state[2] = clip(int(state[2] + dv[0]), a_min=0, a_max=4)
        state[3] = clip(int(state[3] + dv[1]), a_min=0, a_max=4)
        update_state(state, terminated, y_max, x_max, track, value_dict, start_coords, finish_bounds)

        if terminated:
            break
    
    return terminated

fn main() raises -> None:
    # var np = Python.import_module('numpy')
    var value_dict = build_value_dict()
    # read track in from the file
    var file_name = "track1.txt"
    var track = read_track(file_name, value_dict)

    var x_max = track.shape()[1] - 1
    var y_max = track.shape()[0] - 1
    
    # # get the position of the finish - to be used when determining an intersection
    var finish_bounds = get_finish_coords(file_name)

    var state = SIMD[DType.int8, 4](0, 0, 0, 0)
    # For some reason the type system is crying about these being used for the state reset, so it's just hardcoded...
    var start_coords = get_start_coords(file_name)

    # create the policy and Q - it needs to cover the full space and select actions
    # the size of the state space is y, x, 5, 5, 9
    var state_space = (track.shape()[0], track.shape()[1], 5, 5, 9)
    var C = Tensor[DType.float32].rand(TensorShape(state_space))
    var Q = Tensor[DType.float32].randn(TensorShape(state_space), mean=-35)
    var policy = Q.argmax(axis=4).astype[DType.int8]()
    var gamma = 0.9
    var num_episodes = 50_000_000
    var finished_count = 0
    var eps: Float16 = 0.05
    var start = time.now()
    for i in range(num_episodes):

        # print updates
        if i % 250_000 == 0:
            print("Beginning episode", i)

        var states_hist = InlineList[SIMD[DType.int8, 4], 500]()
        var actions_hist = InlineList[Int8, 500]()
        var T = 0
        var terminated = simulate_episode[max_steps=500](policy, state, states_hist, actions_hist, T, y_max, x_max, track, value_dict, start_coords, finish_bounds, eps=eps)
        if not terminated:
            continue
        finished_count += 1

        var G: Float32 = 0
        var W: Float32 = 1

        for t in range(T - 1, -1, -1):
            if t == T - 1:
                pass # R = 0 for terminal reward/action
            else:
                G = gamma * G - 1
            var S_t = states_hist[t]
            var A_t = actions_hist[t]
            var St_At = Index(S_t[0], S_t[1], S_t[2], S_t[3], A_t[4])
            var St = Index(S_t[0], S_t[1], S_t[2], S_t[3])
            C[St_At] += W
            Q[St_At] += W / C[St_At] * (G - Q[St_At])

            # manually slice through Q[S_t] to find the max value
            var max_val: Float32 = -500
            var argmax: Int8 = -1
            for a in range(9):
                var Q_val = Q[Index(S_t[0], S_t[1], S_t[2], S_t[3], a)]
                if Q_val > max_val:
                    max_val = Q_val
                    argmax = i
            policy[St] = argmax
            if A_t != policy[St]:
                break
            W = W * (1 - Float32(eps) + Float32(eps / 9))

    var end = time.now()
    print(num_episodes, 'episodes completed')
    var mins_elapsed = (end - start) / (1_000_000_000)
    # I'd round but mojo's round function doesn't work
    print(int(mins_elapsed / 60), 'minutes taken')
    policy.save('mojo_policy.tens')