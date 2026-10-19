#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  set(module_name ${COLLECTIONS_MODULE_NAME})
else()
  set(module_name OrderedCollections)
  add_library(OrderedCollections
    ${COLLECTIONS_ORDERED_SOURCES})
  target_link_libraries(OrderedCollections PRIVATE
    InternalCollectionsUtilities)
  set_target_properties(OrderedCollections PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  _install_target(OrderedCollections)
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS OrderedCollections)
endif()

target_sources(${module_name} PRIVATE
  "HashTable/_HashTable.swift"
  "HashTable/_HashTable+Bucket.swift"
  "HashTable/_HashTable+BucketIterator.swift"
  "HashTable/_HashTable+Constants.swift"
  "HashTable/_HashTable+CustomStringConvertible.swift"
  "HashTable/_Hashtable+Header.swift"
  "HashTable/_HashTable+Testing.swift"
  "HashTable/_HashTable+UnsafeHandle.swift"
  "OrderedDictionary/OrderedDictionary.swift"
  "OrderedDictionary/OrderedDictionary+Codable.swift"
  "OrderedDictionary/OrderedDictionary+CustomReflectable.swift"
  "OrderedDictionary/OrderedDictionary+Deprecations.swift"
  "OrderedDictionary/OrderedDictionary+Descriptions.swift"
  "OrderedDictionary/OrderedDictionary+Elements.SubSequence.swift"
  "OrderedDictionary/OrderedDictionary+Elements.swift"
  "OrderedDictionary/OrderedDictionary+Equatable.swift"
  "OrderedDictionary/OrderedDictionary+ExpressibleByDictionaryLiteral.swift"
  "OrderedDictionary/OrderedDictionary+Hashable.swift"
  "OrderedDictionary/OrderedDictionary+Initializers.swift"
  "OrderedDictionary/OrderedDictionary+Invariants.swift"
  "OrderedDictionary/OrderedDictionary+Partial MutableCollection.swift"
  "OrderedDictionary/OrderedDictionary+Partial RangeReplaceableCollection.swift"
  "OrderedDictionary/OrderedDictionary+Sendable.swift"
  "OrderedDictionary/OrderedDictionary+Sequence.swift"
  "OrderedDictionary/OrderedDictionary+Sendable.swift"
  "OrderedDictionary/OrderedDictionary+Values.swift"
  "OrderedSet/OrderedSet+Codable.swift"
  "OrderedSet/OrderedSet+CustomReflectable.swift"
  "OrderedSet/OrderedSet+Descriptions.swift"
  "OrderedSet/OrderedSet+Diffing.swift"
  "OrderedSet/OrderedSet+Equatable.swift"
  "OrderedSet/OrderedSet+ExpressibleByArrayLiteral.swift"
  "OrderedSet/OrderedSet+Hashable.swift"
  "OrderedSet/OrderedSet+Initializers.swift"
  "OrderedSet/OrderedSet+Insertions.swift"
  "OrderedSet/OrderedSet+Invariants.swift"
  "OrderedSet/OrderedSet+Partial MutableCollection.swift"
  "OrderedSet/OrderedSet+Partial RangeReplaceableCollection.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra formIntersection.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra formSymmetricDifference.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra formUnion.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra intersection.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isDisjoint.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isEqualSet.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isStrictSubset.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isStrictSuperset.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isSubset.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra isSuperset.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra subtract.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra subtracting.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra symmetricDifference.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra union.swift"
  "OrderedSet/OrderedSet+Partial SetAlgebra+Basics.swift"
  "OrderedSet/OrderedSet+RandomAccessCollection.swift"
  "OrderedSet/OrderedSet+ReserveCapacity.swift"
  "OrderedSet/OrderedSet+Sendable.swift"
  "OrderedSet/OrderedSet+SubSequence.swift"
  "OrderedSet/OrderedSet+Testing.swift"
  "OrderedSet/OrderedSet+UnorderedView.swift"
  "OrderedSet/OrderedSet+UnstableInternals.swift"
  "OrderedSet/OrderedSet.swift"
  "Utilities/_UnsafeBitset.swift"
  )
