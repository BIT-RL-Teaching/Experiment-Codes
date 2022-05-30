#!/usr/bin/python

import numpy as np

class Item(object):
    def __init__(self, pos_x, pos_y):
        self.x = pos_x
        self.y = pos_y

class MovableItem(Item):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.initial_x = pos_x
        self.initial_y = pos_y
    
    def move(self, x, y):
        self.x = x
        self.y = y
    
    def refresh(self):
        self.x = self.initial_x
        self.y = self.initial_y

class Food(MovableItem):
    # 0 for unchoopped 1 for chopped
    def __init__(self, pos_x, pos_y, chopped = False):
        super().__init__(pos_x, pos_y)
        self.chopped = chopped
        self.cur_chopped_times = 0
        self.required_chopped_times = 3        
    
    def chop(self):
        if not self.chopped:
            self.cur_chopped_times += 1
            if self.cur_chopped_times >= self.required_chopped_times:
                self.chopped = True

    def refresh(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.chopped = False
        self.cur_chopped_times = 0

class Tomato(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "tomato"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedTomato"
        else:
            return "FreshTomato"

class Lettuce(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "lettuce"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedLettuce"
        else:
            return "FreshLettuce"

class FixedItem(Item):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y)
        self.holding = holding

    def hold(self, items):
        self.holding = items
    
    def release(self):
        self.holding = None



class Knife(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "knife"
    
    @property
    def name(self):
        return "cutboard"

class Delivery(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "delivery"

    @property
    def name(self):
        return "delivery"


class Plate(MovableItem):
    def __init__(self, pos_x, pos_y, containing = None):
        super().__init__(pos_x, pos_y)
        self.containing = containing
        self.rawName = "plate"
    
    def contain(self, items):
        if self.containing:
            self.containing.append(items)
        else:
            self.containing = [items]
        for item in self.containing:
            item.move(self.x, self.y)
    
    def move(self, x, y):
        super().move(x, y)
        if self.containing:
            for item in self.containing:
                item.move(x, y)

    def release(self):
        self.containing = None

    @property
    def name(self):
        return "plate"

    @property
    def containedName(self):
        if self.containing:
            if len(self.containing) == 1:
                if isinstance(self.containing[0], Tomato):
                    return "ChoppedTomato"
                elif isinstance(self.containing[0], Lettuce):
                    return "ChoppedLettuce"
            elif len(self.containing) == 2:
                    return "ChoppedLettuce-ChoppedTomato"
        else:
            return None


class Agent(MovableItem):
    def __init__(self, pos_x, pos_y, holding = None, color = None):
        super().__init__(pos_x, pos_y)
        self.holding = holding
        self.color = color
        self.moved = False
        self.obs = None
        self.pomap = None
        self.rawName = "agent"

    def pickup(self,item):
        self.holding = item
        item.move(self.x, self.y)
    
    def putdown(self, x, y):
        self.holding.move(x, y)
        self.holding = None

    def move(self, x, y):
        super().move(x, y)
        self.moved = True
        if self.holding:
            self.holding.move(x, y)


