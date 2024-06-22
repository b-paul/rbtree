//! This crate implements a Red-Black tree. This is mainly being made for learning/as an exercise.

#![no_std]
#![deny(missing_docs)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::Hash;
use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ptr::NonNull;

/// A Red-Black tree for use as an ordered map. See the root level documentation for more info.
pub struct RbTree<K: Ord, V> {
    /// The root node of the tree.
    root: Option<NonNull<RbNode<K, V>>>,
    /// The amount of nodes in the tree.
    len: usize,
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
        self.len += 1;
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
        self.len -= 1;

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
                            let child = (*sibling.as_ptr())[sibling_direction.opposite()]
                                .unwrap_unchecked();
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

    /// Returns the amount of elements stored in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create a new empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a borrowing iterator over the key value pairs in the tree.
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            stack: Vec::new(),
            cur: self.root,
            len: self.len,
            _lifetime: PhantomData,
        }
    }

    /// Return an iterator over pairs of keys and mutable references to values in the tree.
    pub fn iter_mut(&self) -> IterMut<K, V> {
        IterMut {
            stack: Vec::new(),
            cur: self.root,
            len: self.len,
            _lifetime: PhantomData,
        }
    }
}

impl<K: Ord, V> Drop for RbTree<K, V> {
    fn drop(&mut self) {
        let mut stack = Vec::new();
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

impl<K: Ord, V> Default for RbTree<K, V> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}

impl<K: Ord + Clone, V: Clone> Clone for RbTree<K, V> {
    fn clone(&self) -> Self {
        Self::from_iter(self.iter().map(|(k, v)| (k.clone(), v.clone())))
    }
}

impl<K: Ord + Debug, V: Debug> Debug for RbTree<K, V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_map().entries(self).finish()
    }
}

impl<K: Ord + PartialEq, V: PartialEq> PartialEq for RbTree<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<K: Ord + Eq, V: Eq> Eq for RbTree<K, V> {}

impl<K: Ord, V> FromIterator<(K, V)> for RbTree<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut tree = RbTree::default();
        for (k, v) in iter {
            tree.insert(k, v);
        }
        tree
    }
}

impl<'a, K: Ord + Copy, V: Copy> Extend<(&'a K, &'a V)> for RbTree<K, V> {
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        for (&k, &v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Ord, V: PartialOrd> PartialOrd for RbTree<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

impl<K: Ord, V: Ord> Ord for RbTree<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other)
    }
}

impl<K: Ord + Hash, V: Hash> Hash for RbTree<K, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for (k, v) in self {
            (k, v).hash(state);
        }
    }
}

impl<K: Ord, V> Extend<(K, V)> for RbTree<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a RbTree<K, V> {
    type Item = (&'a K, &'a V);

    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Ord, V> IntoIterator for RbTree<K, V> {
    type Item = (K, V);

    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let tree = ManuallyDrop::new(self);
        IntoIter {
            stack: Vec::new(),
            cur: tree.root,
            len: tree.len,
        }
    }
}

/// A borrowing iterator over the elements of an `RbTree`.
///
/// The order of elements that the iterator generates is close enough to random I guess.
pub struct Iter<'a, K: Ord, V> {
    stack: Vec<NonNull<RbNode<K, V>>>,
    cur: Option<NonNull<RbNode<K, V>>>,
    len: usize,
    _lifetime: PhantomData<&'a V>,
}

impl<'a, K: Ord + 'a, V: 'a> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;

        for child in unsafe { (*cur.as_ptr()).child.into_iter().flatten() } {
            self.stack.push(child);
        }
        self.cur = self.stack.pop();

        self.len -= 1;

        unsafe { Some((&(*cur.as_ptr()).key, &(*cur.as_ptr()).val)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K: Ord + 'a, V: 'a> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K: Ord + 'a, V: 'a> FusedIterator for Iter<'a, K, V> {}

/// An iterator over mutable references of the elements of an `RbTree`.
///
/// The order of elements that the iterator generates is close enough to random I guess.
pub struct IterMut<'a, K: Ord, V> {
    stack: Vec<NonNull<RbNode<K, V>>>,
    cur: Option<NonNull<RbNode<K, V>>>,
    len: usize,
    _lifetime: PhantomData<&'a V>,
}

impl<'a, K: Ord + 'a, V: 'a> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;

        for child in unsafe { (*cur.as_ptr()).child.into_iter().flatten() } {
            self.stack.push(child);
        }
        self.cur = self.stack.pop();

        self.len -= 1;

        unsafe { Some((&(*cur.as_ptr()).key, &mut (*cur.as_ptr()).val)) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K: Ord + 'a, V: 'a> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, K: Ord + 'a, V: 'a> FusedIterator for IterMut<'a, K, V> {}

/// An owning iterator over the elements in an `RbTree`.
///
/// The order of elements that the iterator generates is close enough to random I guess.
pub struct IntoIter<K: Ord, V> {
    stack: Vec<NonNull<RbNode<K, V>>>,
    cur: Option<NonNull<RbNode<K, V>>>,
    len: usize,
}

impl<K: Ord, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;

        for child in unsafe { (*cur.as_ptr()).child.into_iter().flatten() } {
            self.stack.push(child);
        }
        self.cur = self.stack.pop();

        let cur = unsafe { Box::from_raw(cur.as_ptr()) };
        self.len -= 1;

        Some((cur.key, cur.val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<K: Ord, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<K: Ord, V> FusedIterator for IntoIter<K, V> {}

impl<K: Ord, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        for _ in self {}
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
    fn test_ops() {
        let mut tree = RbTree::default();

        const COUNT: usize = 1000000;

        for key in 0..COUNT {
            tree.insert(key, key);
        }
        for key in 0..COUNT {
            tree.insert(key, key);
        }
        assert!(tree.len() == COUNT);
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

    #[test]
    fn test_iter() {
        let mut tree = RbTree::default();

        const COUNT: usize = 100;

        for key in 0..COUNT {
            tree.insert(key, key);
        }

        for (key, val) in tree.iter() {
            assert_eq!(key, val);
        }

        for (key, val) in tree.iter_mut() {
            assert_eq!(key, val);
            *val += 1;
        }

        for (key, val) in tree {
            assert_eq!(key + 1, val);
        }
    }

    #[test]
    fn test_traits() {
        let mut tree = RbTree::from_iter((0..100).map(|x| (x, x)));

        let mut tree2 = RbTree::new();
        for x in 0..100 {
            tree2.insert(x, x);
        }
        assert_eq!(tree, tree2);

        tree.extend(tree2.into_iter().map(|(k, v)| (k + 100, v)));
        for x in 100..200 {
            assert!(tree.get(&x) == Some(&(x - 100)));
        }

        let tree2 = tree.clone();
        for item in tree2.iter() {
            assert!(tree.iter().any(|i| i == item));
        }
        for item in tree.iter() {
            assert!(tree2.iter().any(|i| i == item));
        }
    }
}
