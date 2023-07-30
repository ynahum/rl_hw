import os
import copy

g_unique_valids_list = []

class State:
    def __init__(self, s=None):
        if s is not None:
            self._array = s
            self._c = len(self._array[0])
            self._r = len(self._array)
            self._valids_locs_dict = {}
            temp_list = []
            for i in range(self._r):
                for j in range(self._c):
                    num = self._array[i][j]
                    if num > 0:
                        if self._valids_locs_dict.get(num) is not None:
                            self._valids_locs_dict[num].append((i, j))
                        else:
                            self._valids_locs_dict[num] = []
                        temp_list.append(num)
                    self._valids_locs_dict[num] = sorted(self._valids_locs_dict[num])
            g_unique_valids_list = sorted(set(temp_list))

    def copy(self):
        result = State()
        result._array = copy.deepcopy(self._array)
        result._c = self._c
        result._r = self._r
        result._valids_locs_dict = copy.deepcopy(self._valids_locs_dict)
        return result

    def _get_location_of_num(self, c):
        locations = []
        for i in range(self._r):
            for j in range(self._c):
                if self._array[i][j] == c:
                    locations.append((i, j))
        return locations

    def _get_empty_locations(self):
        return self._get_location_of_num(0)

    def to_string(self):
        return os.linesep.join([' '.join(m) for m in self._array])

    def __eq__(self, other):
        return self.to_string() == other.to_string()

    def __lt__(self, other):
        return self.to_string() < other.to_string()

    def can_it_go_direction(self, num, direction):
        for location in self._valids_locs_dict[num]:
            row = location[0]
            col = location[1]
            if direction == 'U':
                if row == 0 or self._array[row - 1] != 0:
                    return False
            if direction == 'D':
                if row == (self._r-1) or self._array[row + 1] != 0:
                    return False
            if direction == 'L':
                if col == 0 or self._array[row][col - 1] != 0:
                    return False
            if direction == 'R':
                if col == (self._c-1) or self._array[row][col + 1] != 0:
                    return False
        return True

    def get_actions(self):
        actions = []
        for unique_valid in g_unique_valids_list:
            if self.can_it_go_direction('U'):
                actions.append((unique_valid, 'U'))
            if self.can_it_go_direction('D'):
                actions.append((unique_valid, 'D'))
            if self.can_it_go_direction('L'):
                actions.append((unique_valid, 'L'))
            if self.can_it_go_direction('R'):
                actions.append((unique_valid, 'R'))

        return actions

    def apply_action(self, a):

        valid_actions = self.get_actions()
        assert a in valid_actions

        new_state = self.copy()

        action_num = a[0]
        action_dir = a[1]

        new_locs = []
        for location in self._valids_locs_dict[action_num]:
            row = location[0]
            col = location[1]
            new_row = row
            new_col = col
            if action_dir == 'U':
                new_row = row - 1
            if action_dir == 'D':
                new_row = row + 1
            if action_dir == 'L':
                new_col = col - 1
            if action_dir == 'R':
                new_col = col + 1
            self._array[new_row][new_col] = action_num
            self._array[row][col] = 0
            new_locs.append((row, col))

        self._valids_locs_dict[action_num] = sorted(new_locs)

        return new_state

    def get_manhattan_distance(self, other):
        total_distance = 0
        for unique_valid in g_unique_valids_list:
            for idx, self_location in enumerate(self._valids_locs_dict):
                other_location = other._valids_locs_dict[idx]
                diff = abs(self_location[0] - other_location[0]) + abs(self_location[1] - other_location[1])
                total_distance += diff
        return total_distance

    def is_same(self, other):
        return self.get_manhattan_distance(other) == 0


if __name__ == '__main__':
    initial_state = State()
    print('initial state')
    print(initial_state.to_string())
    initial_actions = initial_state.get_actions()
    print('actions: {}'.format(initial_actions))
    down_state = initial_state.apply_action('d')
    print('distance to self:')
    print(initial_state.get_manhattan_distance(initial_state))
    print('one down from initial')
    print(down_state.to_string())
    print('distance between both:')
    print(down_state.get_manhattan_distance(initial_state))
    right_state = initial_state.apply_action('r')
    print('one to the right from initial')
    print(right_state.to_string())
