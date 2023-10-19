#! /usr/bin/env python3

"""tng.py: A Python module for the TNG model."""

from icecream import ic
import numpy as np


class UidMap(object):

    """A class for the UidMap object."""

    def __init__(self, initial_population_size, number_of_timesteps, max_children=8, year_size=256):

        """Constructor for the UidMap class."""

        self.pop_bits = np.uint32(np.ceil(np.log2(initial_population_size)))
        self.time_bits = np.uint32(np.ceil(np.log2(number_of_timesteps)))
        self.child_bits = np.uint32(np.ceil(np.log2(max_children)))
        self.day_bits = np.uint32(np.ceil(np.log2(year_size)))
        self.year_bits = self.time_bits - self.day_bits

        self.ssn_offset = self.child_bits + self.time_bits
        self.child_offset = self.time_bits
        self.child_mask = np.uint64((1 << self.child_bits) - 1)
        self.time_mask = np.uint64((1 << self.time_bits) - 1)
        self.day_mask = np.uint64((1 << self.day_bits) - 1)

        num_years = np.uint32(1 << self.year_bits)
        self.mapping = [ None for _ in range(num_years) ]

        return
    
    def make_uid(self, ssn, time, child):

        """Make a UID from its constituent parts."""

        return np.uint64((ssn << self.ssn_offset) + (child << self.child_offset) + time)
    
    def parse_uid(self, uid):

        """Parse a UID into its constituent parts."""

        uid = np.uint64(uid)
        ssn = uid >> self.ssn_offset    # left most bits
        child = (uid >> self.child_offset) & self.child_mask    # after ssn bits
        time = uid & self.time_mask # right most bits
        year = time >> self.day_bits    # year is time modulo year_size
        day = time & self.day_mask  # day is time remainder year_size

        return (ssn, child, day, year)
    
    def add_individual(self, uid, index):

        """Add an individual to the population."""

        # year    child   day    ssn
        # (list)  (list)  (list)  (numpy array)
        # |-----| |-----| |-----| |-----|
        # |  0  | |  0  | |  0  | |  0  |
        # | ... | | ... | | ... | | ... |
        # | max | | max | | max | | max |
        # | year| |child| | day | | ssn |
        # |-----| |-----| |-----| |-----|

        uid = np.uint64(uid)
        ssn, child, day, year = self.parse_uid(uid)

        ic(uid, ssn, child, day, year)

        if self.mapping[year] is None:  # we don't have a table for this year yet
            # Allocate a new child table in this year
            ic(f"Didn't have a table for year {year}")
            self.mapping[year] = [ None for _ in range(1 << self.child_bits) ]

        child_list = self.mapping[year]

        if child_list[child] is None:  # we don't have a table for this child yet
            # Allocate a new day table for this child
            ic(f"Didn't have a table for child {child}")
            child_list[child] = [ None for _ in range(1 << self.day_bits) ]

        day_list = child_list[child]

        if day_list[day] is None:  # we don't have a table for this day yet
            # Allocate a new SSN table for this day
            ic(f"Didn't have a table for day {day}")
            day_list[day] = np.zeros((1 << self.pop_bits, 2), dtype=np.uint64)
            day_list[day][:,0] = 0xFFFFFFFFFFFFFFFF # initialize all indices to -1

        ssn_array = day_list[day]
        ssn_array[ssn,0] = index
        ssn_array[ssn,1] = uid

        return
    
    def get_index(self, uid):

        """Get the index of an individual."""

        uid = np.uint64(uid)
        ssn, child, day, year = self.parse_uid(uid)
        ic(uid, ssn, child, day, year)
        return self.mapping[year][child][day][ssn,0]
    
    def uids(self):

        """Get the UIDs of all individuals."""

        for year, child_list in enumerate(self.mapping):
            if child_list is None:
                continue
            for day, day_list in enumerate(child_list):
                if day_list is None:
                    continue
                for ssn_array in day_list:
                    if ssn_array is None:
                        continue
                    for i in range(len(ssn_array)):
                        index, uid = ssn_array[i,0], ssn_array[i,1]
                        if index != 0xFFFFFFFFFFFFFFFF:
                            yield uid
    
    def __str__(self) -> str:

        """Return a string representation of the UidMap object."""

        return f"UidMap: pop_bits={self.pop_bits}, time_bits={self.time_bits}, child_bits={self.child_bits}, year_bits={self.year_bits}"
    

def main():

    """Main function for tng.py."""

    population_size = 10_000
    simulation_duration = 365*20

    mapping = UidMap(population_size, simulation_duration)
    print(mapping)

    for i in range(32):
        mapping.add_individual(mapping.make_uid(i, 0, 0), i)

    # for uid in mapping.uids():
    #     index = mapping.get_index(uid)
    #     print(uid, index)
    #     if index % 2 == 0:
    #         ssn, _child, _day, _year = mapping.parse_uid(uid)
    #         t = np.random.randint(365*20)
    #         mapping.add_individual(mapping.make_uid(ssn, t, 1), index)

    print("Adding children...")
    for i in range(8):
        ssn = np.random.randint(population_size)
        time = np.random.randint(simulation_duration)
        child = np.random.randint(8)
        print(f"Adding child {i+32} with ssn={ssn}, time={time}, child={child}")
        uid = mapping.make_uid(ssn, time, child)
        mapping.add_individual(uid, i+32)

    for uid in mapping.uids():
        index = mapping.get_index(uid)
        print(uid, index)

    return


if __name__ == '__main__':
    main()
