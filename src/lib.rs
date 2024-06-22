//! This crate implements a Red-Black tree. This is mainly being made for learning/as an exercise.

#![no_std]
#![deny(missing_docs)]

extern crate alloc;

use alloc::boxed::Box;
use core::cmp::Ordering;
use core::marker::PhantomData;
use core::ptr::NonNull;

/// A Red-Black tree for use as an ordered map. See the root level documentation for more info.
#[derive(Default)]
pub struct RbTree<K: Ord, V> {
    /// The root node of the tree.
    root: Option<NonNull<RbNode<K, V>>>,

    _key_marker: PhantomData<K>,
    _val_marker: PhantomData<V>,
}

impl<K: Ord, V> RbTree<K, V> {
    /// Rotate a subtree of the tree in a direction.
    ///
    /// ## Safety
    /// - All pointers must be either valid or null
    /// - The child in the opposite direction of dir of node must be non null.
    unsafe fn rotate_dir_root(
        &mut self,
        root: NonNull<RbNode<K, V>>,
        dir: Direction,
    ) -> NonNull<RbNode<K, V>> {
        let parent = (*root.as_ptr()).parent;
        let opposite = (*root.as_ptr())[dir.opposite()];
        debug_assert!(opposite.is_some());
        let opposite = opposite.unwrap_unchecked();
        let middle = (*opposite.as_ptr())[dir];

        (*root.as_ptr())[dir.opposite()] = middle;
        if let Some(middle) = middle {
            (*middle.as_ptr()).parent = Some(root);
        }
        (*opposite.as_ptr())[dir] = Some(root);
        (*root.as_ptr()).parent = Some(opposite);
        (*opposite.as_ptr()).parent = parent;
        match parent {
            Some(parent) => {
                let parent_dir = match (*parent.as_ptr())[Direction::Left] == Some(root) {
                    true => Direction::Left,
                    false => Direction::Right,
                };
                (*parent.as_ptr())[parent_dir] = Some(opposite);
            }
            None => self.root = Some(opposite),
        }

        opposite
    }

    /// Replace a node in the tree with a new node. The caller is responsible for freeing all
    /// memory after this operation.
    ///
    /// ## Safety
    /// - All pointers ust be either valid or null
    unsafe fn transplant(
        &mut self,
        point: NonNull<RbNode<K, V>>,
        new: Option<NonNull<RbNode<K, V>>>,
    ) {
        match (*point.as_ptr()).parent {
            None => {
                self.root = new;
            }
            Some(parent) => {
                if Some(point) == (*parent.as_ptr())[Direction::Left] {
                    (*parent.as_ptr())[Direction::Left] = new;
                } else {
                    (*parent.as_ptr())[Direction::Right] = new;
                }
            }
        }
        if let Some(new) = new {
            (*new.as_ptr()).parent = (*point.as_ptr()).parent;
        }
    }

    fn min_node(&self, mut root: Option<NonNull<RbNode<K, V>>>) -> Option<NonNull<RbNode<K, V>>> {
        while let Some(node) = root {
            root = unsafe { (*node.as_ptr())[Direction::Left] };
        }
        root
    }

    /// Get a value if it exists.
    ///
    /// ```rust
    /// # use rbtree::RbTree;
    ///
    /// let mut map = RbTree::default();
    ///
    /// map.insert(4, 6);
    /// map.insert(5, 7);
    /// map.insert(6, 8);
    ///
    /// assert!(map.get(&4) == Some(&6));
    /// assert!(map.get(&5) == Some(&7));
    /// assert!(map.get(&6) == Some(&8));
    /// ```
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut cur = self.root;

        // SAFETY: all pointers should be kept valid in this data structure.
        while let Some(node) = cur {
            let (node_key, left, right) = unsafe {
                (
                    &(*node.as_ptr()).key,
                    (*node.as_ptr()).child[0],
                    (*node.as_ptr()).child[1],
                )
            };
            match key.cmp(node_key) {
                Ordering::Less => cur = left,
                Ordering::Equal => break,
                Ordering::Greater => cur = right,
            }
        }
        cur.map(|n| unsafe { &(*n.as_ptr()).val })
    }

    /// Insert a key value pair into the map.
    ///
    /// ```rust
    /// # use rbtree::RbTree;
    ///
    /// let mut map = RbTree::default();
    ///
    /// map.insert(4, 6);
    /// map.insert(5, 7);
    /// map.insert(4, 8);
    ///
    /// assert!(map.get(&5) == Some(&7));
    /// assert!(map.get(&4) == Some(&8));
    /// ```
    pub fn insert(&mut self, key: K, val: V) {
        let mut parent = None;
        let mut cur = self.root;
        let mut dir = Direction::Left;
        // Traverse the BST to find the place to insert our key/value node. If a node with the same
        // key as us already exists, replace its value and return. It is important that we keep
        // track of the parent and direction from the parent that the current node is in so that we
        // can update the parent node, and assign the parent to our new node correctly.
        while let Some(node) = cur {
            let (node_key, left, right) = unsafe {
                (
                    &(*node.as_ptr()).key,
                    (*node.as_ptr()).child[0],
                    (*node.as_ptr()).child[1],
                )
            };
            match key.cmp(node_key) {
                Ordering::Less => {
                    parent = cur;
                    cur = left;
                    dir = Direction::Left;
                }
                Ordering::Equal => {
                    unsafe { (*node.as_ptr()).val = val };
                    return;
                }
                Ordering::Greater => {
                    parent = cur;
                    cur = right;
                    dir = Direction::Right;
                }
            }
        }

        // cur must now be a None node that we want to insert into. We now replace it with a new
        // node containing our key value pair. We initially set the colour to red and will correct
        // this afterwards if needed.
        let node = Box::new(RbNode {
            key,
            val,
            colour: Colour::Red,
            parent,
            child: [None, None],
        });
        cur = unsafe { Some(NonNull::new_unchecked(Box::into_raw(node))) };
        match parent {
            None => {
                self.root = cur;
                return;
            }
            Some(parent) => {
                unsafe { (*parent.as_ptr())[dir] = cur };
            }
        }

        while let Some(node) = cur {
            let Some(parent) = (unsafe { (*node.as_ptr()).parent }) else {
                break;
            };
            if unsafe { (*parent.as_ptr()).colour == Colour::Black } {
                break;
            }
            debug_assert!(unsafe { (*node.as_ptr()).colour == Colour::Red });
            let grandparent = unsafe { (*parent.as_ptr()).parent };
            // First we check whether we have a grandparent. If we don't have a grandparent, our
            // parent will be the root node, and we can just correct it to be black after exiting
            // the loop.
            match grandparent {
                None => {
                    break;
                }
                Some(grandparent) => {
                    // We obtain the direction that our parent came from in order to compute find
                    // the uncle, and to perform some rotations later.
                    let grandparent_dir =
                        match Some(parent) == unsafe { (*grandparent.as_ptr())[Direction::Left] } {
                            true => Direction::Left,
                            false => Direction::Right,
                        };
                    let uncle = unsafe { (*grandparent.as_ptr())[grandparent_dir.opposite()] };
                    if uncle.is_some_and(|u| unsafe { (*u.as_ptr()).colour == Colour::Red }) {
                        // If the uncle's colour is red (noting that None nodes are black), we
                        // recolour the uncle and the parent to black, and set the grandparent to
                        // red. The grandparent must be black since the uncle is red.
                        unsafe {
                            (*uncle.unwrap_unchecked().as_ptr()).colour = Colour::Black;
                            (*parent.as_ptr()).colour = Colour::Black;
                            (*grandparent.as_ptr()).colour = Colour::Red;
                        }
                        // Set the next cur to be the grandparent.
                        cur = Some(grandparent);
                    } else {
                        // We want to rotate the tree to give us a /\ triangle shape, correcting
                        // for colouring.
                        if Some(node) == unsafe { (*parent.as_ptr())[grandparent_dir.opposite()] } {
                            // We have a zig-zag shape. We can rotate this into a zig zig shape to
                            // get a nice rotation.
                            cur = Some(parent);
                            unsafe {
                                self.rotate_dir_root(parent, grandparent_dir);
                                (*node.as_ptr()).colour = Colour::Black;
                            }
                        } else {
                            unsafe { (*parent.as_ptr()).colour = Colour::Black };
                        }
                        // We now have a zig zig shape, and we can then rotate it to give a
                        // triangle. We have coloured the node that sits in the top of the triangle
                        // as black, and we set the grandparent to keep the black height property.
                        unsafe {
                            self.rotate_dir_root(grandparent, grandparent_dir.opposite());
                            (*grandparent.as_ptr()).colour = Colour::Red;
                        }
                    }
                }
            }
        }
        // Finally we correct the root node to be black.
        if let Some(r) = self.root {
            unsafe { (*r.as_ptr()).colour = Colour::Black }
        }
    }

    /// Remove a value from the map (if it exists).
    ///
    /// ```rust
    /// # use rbtree::RbTree;
    ///
    /// let mut map = RbTree::default();
    ///
    /// map.insert(4, 6);
    /// assert!(map.remove(&4) == Some(6));
    /// assert!(map.get(&4) == None);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        // Find the node that we want to remove
        let mut cur = self.root;
        while let Some(node) = cur {
            let (node_key, left, right) = unsafe {
                (
                    &(*node.as_ptr()).key,
                    (*node.as_ptr()).child[0],
                    (*node.as_ptr()).child[1],
                )
            };
            match key.cmp(node_key) {
                Ordering::Less => cur = left,
                Ordering::Equal => break,
                Ordering::Greater => cur = right,
            }
        }
        // If cur is None, we didn't find a node with our key in the tree, so there is nothing to
        // remove.
        let node = cur?;

        let mut replacement;
        let mut replacement_parent;
        let mut old_colour = unsafe { (*node.as_ptr()).colour };

        // If either side of the node that we are removing is None, we can just replace our node
        // with the other side.
        if unsafe { (*node.as_ptr())[Direction::Left].is_none() } {
            unsafe {
                replacement = (*node.as_ptr())[Direction::Right];
                replacement_parent = (*node.as_ptr()).parent;
                self.transplant(node, replacement);
            }
        } else if unsafe { (*node.as_ptr())[Direction::Right].is_none() } {
            unsafe {
                replacement = (*node.as_ptr())[Direction::Left];
                replacement_parent = (*node.as_ptr()).parent;
                self.transplant(node, replacement);
            }
        } else {
            // If both sides have actual contents, then we can replace the node we are at with the
            // minimum node in the right subtree.
            unsafe {
                let min = self.min_node((*node.as_ptr())[Direction::Right]);
                // We should have already accounted for the case where the right side is None.
                debug_assert!(min.is_some());
                let min = min.unwrap_unchecked();
                old_colour = (*min.as_ptr()).colour;
                replacement = (*min.as_ptr())[Direction::Right];
                replacement_parent = Some(min);
                if (*min.as_ptr()).parent != cur {
                    self.transplant(min, (*min.as_ptr())[Direction::Right]);
                    (*min.as_ptr())[Direction::Right] = (*node.as_ptr())[Direction::Right];
                    (*(*min.as_ptr())[Direction::Right]
                        .unwrap_unchecked()
                        .as_ptr())
                    .parent = Some(min);
                }
                self.transplant(node, Some(min));
                (*min.as_ptr())[Direction::Left] = (*node.as_ptr())[Direction::Left];
                (*(*min.as_ptr())[Direction::Left].unwrap_unchecked().as_ptr()).parent = Some(min);
                (*min.as_ptr()).colour = (*node.as_ptr()).colour;
            }
        }
        let removed = *unsafe { Box::from_raw(node.as_ptr()) };

        if old_colour == Colour::Black {
            while let Some(parent) = replacement_parent {
                unsafe {
                    debug_assert!((*parent.as_ptr()).colour == Colour::Black);
                    if replacement.is_some_and(|n| (*n.as_ptr()).colour == Colour::Red) {
                        break;
                    }
                    let sibling_direction = match replacement == (*parent.as_ptr())[Direction::Left]
                    {
                        true => Direction::Right,
                        false => Direction::Left,
                    };
                    let sibling = (*parent.as_ptr())[sibling_direction];
                    // Since old_colour is black, we know that in the first iteration, the sibling is
                    // not None. In the next iterations, it also can't be None, since the current node
                    // is black and not None, so by the black height property its sibling cannot have
                    // black height 0.
                    debug_assert!(sibling.is_some());
                    let mut sibling = sibling.unwrap_unchecked();
                    if (*sibling.as_ptr()).colour == Colour::Red {
                        (*sibling.as_ptr()).colour = Colour::Black;
                        (*parent.as_ptr()).colour = Colour::Red;
                        self.rotate_dir_root(parent, sibling_direction.opposite());
                        // Because the replacement is black, the sibling must have non-None
                        // children, else the black height property would be violated.
                        // Additionally, both of its children must be not red, i.e. black.
                        sibling = (*parent.as_ptr())[sibling_direction].unwrap_unchecked();
                    }

                    // From this point, we know that the sibling is black.
                    if (*sibling.as_ptr()).child.into_iter().all(|c| {
                        c.is_none() || c.is_some_and(|n| (*n.as_ptr()).colour == Colour::Black)
                    }) {
                        // If both of the sibling's children are black, we can recolour the sibling
                        // itself to be red, and this will correct the black height error. We then
                        // move up the tree.
                        (*sibling.as_ptr()).colour = Colour::Red;
                        replacement = Some(parent);
                        replacement_parent = (*parent.as_ptr()).parent;
                    } else {
                        if (*sibling.as_ptr())[sibling_direction.opposite()]
                            .is_some_and(|n| (*n.as_ptr()).colour == Colour::Red)
                        {
                            // Lol this unwrap_unchecked is so sad I want if let chains!
                            let child = (*sibling.as_ptr())[sibling_direction.opposite()].unwrap_unchecked();
                            // If the inner child of the sibling is red, then we can rotate it and
                            // recolour so that the outer child is red instead.
                            (*child.as_ptr()).colour = Colour::Black;
                            (*sibling.as_ptr()).colour = Colour::Red;
                            self.rotate_dir_root(sibling, sibling_direction);
                            sibling = child;
                        }
                        // The outer child of the subling is red. We can rotate the subtree about
                        // the parent, colour it black, and recolour the red sibling black so that
                        // the original sid of the new parent after the rotation has one extra
                        // black node than the sibling's side. This means that the tree has been
                        // rebalanced.
                        (*sibling.as_ptr()).colour = (*parent.as_ptr()).colour;
                        (*parent.as_ptr()).colour = Colour::Black;
                        if let Some(child) = (*sibling.as_ptr())[sibling_direction] {
                            (*child.as_ptr()).colour = Colour::Black;
                        }
                        self.rotate_dir_root(parent, sibling_direction.opposite());
                        replacement = self.root;
                        replacement_parent = None;
                    }
                }
            }
            if let Some(node) = replacement {
                unsafe { (*node.as_ptr()).colour = Colour::Black };
            }
        }
        Some(removed.val)
    }
}

impl<K: Ord, V> Drop for RbTree<K, V> {
    fn drop(&mut self) {
        let mut stack = alloc::vec::Vec::new();
        if let Some(root) = self.root {
            stack.push(root);
        }
        while let Some(node) = stack.pop() {
            for child in unsafe { (*node.as_ptr()).child }.into_iter().flatten() {
                stack.push(child);
            }

            drop(unsafe { Box::from_raw(node.as_ptr()) });
        }
    }
}

/// A direction for a node to be in, in a binary tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

impl Direction {
    /// Get the opposite of a direction.
    fn opposite(self) -> Direction {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }
}

/// The colour of an RbNode. See `RbNode` for more info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Colour {
    Red,
    Black,
}

/// An RbNode. Each node has a key value pair, and a colour used for balancing.
struct RbNode<K: Ord, V> {
    key: K,
    val: V,
    /// The colour of this node. Every node has a colour for balancing purposes.
    colour: Colour,

    /// A raw pointer to the parent of this node. This pointer is None iff the node is the root
    /// node.
    parent: Option<NonNull<RbNode<K, V>>>,
    /// Our two child nodes.
    child: [Option<NonNull<RbNode<K, V>>>; 2],
}

impl<K: Ord, V> core::ops::Index<Direction> for RbNode<K, V> {
    type Output = Option<NonNull<RbNode<K, V>>>;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::Left => &self.child[0],
            Direction::Right => &self.child[1],
        }
    }
}

impl<K: Ord, V> core::ops::IndexMut<Direction> for RbNode<K, V> {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::Left => &mut self.child[0],
            Direction::Right => &mut self.child[1],
        }
    }
}

#[cfg(test)]
mod test {
    use crate::RbTree;

    #[test]
    fn test() {
        let mut tree = RbTree::default();

        const COUNT: usize = 1000000;

        for key in 0..COUNT {
            tree.insert(key, key);
        }
        for key in 0..COUNT {
            tree.insert(key, key);
        }
        for key in 0..COUNT {
            assert!(tree.get(&key) == Some(&key));
        }
        for key in 0..COUNT {
            assert!(tree.remove(&key) == Some(key));
        }
        for key in 0..COUNT {
            assert!(tree.get(&key) == None);
        }
    }
}
