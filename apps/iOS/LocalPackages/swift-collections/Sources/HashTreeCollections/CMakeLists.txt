#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2022 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  set(module_name ${COLLECTIONS_MODULE_NAME})
else()
  set(module_name HashTreeCollections)
  add_library(HashTreeCollections
    ${COLLECTIONS_HASHTREE_SOURCES})
  target_link_libraries(HashTreeCollections PRIVATE
    InternalCollectionsUtilities)
  set_target_properties(HashTreeCollections PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  _install_target(HashTreeCollections)
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS HashTreeCollections)
endif()

target_sources(${module_name} PRIVATE
  "HashNode/_AncestorHashSlots.swift"
  "HashNode/_Bitmap.swift"
  "HashNode/_Bucket.swift"
  "HashNode/_Hash.swift"
  "HashNode/_HashLevel.swift"
  "HashNode/_HashNode+Builder.swift"
  "HashNode/_HashNode+Debugging.swift"
  "HashNode/_HashNode+Initializers.swift"
  "HashNode/_HashNode+Invariants.swift"
  "HashNode/_HashNode+Lookups.swift"
  "HashNode/_HashNode+Primitive Insertions.swift"
  "HashNode/_HashNode+Primitive Removals.swift"
  "HashNode/_HashNode+Primitive Replacement.swift"
  "HashNode/_HashNode+Storage.swift"
  "HashNode/_HashNode+Structural compactMapValues.swift"
  "HashNode/_HashNode+Structural filter.swift"
  "HashNode/_HashNode+Structural intersection.swift"
  "HashNode/_HashNode+Structural isDisjoint.swift"
  "HashNode/_HashNode+Structural isEqualSet.swift"
  "HashNode/_HashNode+Structural isSubset.swift"
  "HashNode/_HashNode+Structural mapValues.swift"
  "HashNode/_HashNode+Structural merge.swift"
  "HashNode/_HashNode+Structural subtracting.swift"
  "HashNode/_HashNode+Structural symmetricDifference.swift"
  "HashNode/_HashNode+Structural union.swift"
  "HashNode/_HashNode+Subtree Insertions.swift"
  "HashNode/_HashNode+Subtree Modify.swift"
  "HashNode/_HashNode+Subtree Removals.swift"
  "HashNode/_HashNode+UnsafeHandle.swift"
  "HashNode/_HashNode.swift"
  "HashNode/_HashSlot.swift"
  "HashNode/_HashStack.swift"
  "HashNode/_HashTreeIterator.swift"
  "HashNode/_HashTreeStatistics.swift"
  "HashNode/_RawHashNode+UnsafeHandle.swift"
  "HashNode/_RawHashNode.swift"
  "HashNode/_HashNodeHeader.swift"
  "HashNode/_UnmanagedHashNode.swift"
  "HashNode/_UnsafePath.swift"
  "TreeDictionary/TreeDictionary+Codable.swift"
  "TreeDictionary/TreeDictionary+Collection.swift"
  "TreeDictionary/TreeDictionary+CustomReflectable.swift"
  "TreeDictionary/TreeDictionary+Debugging.swift"
  "TreeDictionary/TreeDictionary+Descriptions.swift"
  "TreeDictionary/TreeDictionary+Equatable.swift"
  "TreeDictionary/TreeDictionary+ExpressibleByDictionaryLiteral.swift"
  "TreeDictionary/TreeDictionary+Filter.swift"
  "TreeDictionary/TreeDictionary+Hashable.swift"
  "TreeDictionary/TreeDictionary+Initializers.swift"
  "TreeDictionary/TreeDictionary+Keys.swift"
  "TreeDictionary/TreeDictionary+MapValues.swift"
  "TreeDictionary/TreeDictionary+Merge.swift"
  "TreeDictionary/TreeDictionary+Sendable.swift"
  "TreeDictionary/TreeDictionary+Sequence.swift"
  "TreeDictionary/TreeDictionary+Values.swift"
  "TreeDictionary/TreeDictionary.swift"
  "TreeSet/TreeSet+Codable.swift"
  "TreeSet/TreeSet+Collection.swift"
  "TreeSet/TreeSet+CustomReflectable.swift"
  "TreeSet/TreeSet+Debugging.swift"
  "TreeSet/TreeSet+Descriptions.swift"
  "TreeSet/TreeSet+Equatable.swift"
  "TreeSet/TreeSet+ExpressibleByArrayLiteral.swift"
  "TreeSet/TreeSet+Extras.swift"
  "TreeSet/TreeSet+Filter.swift"
  "TreeSet/TreeSet+Hashable.swift"
  "TreeSet/TreeSet+Sendable.swift"
  "TreeSet/TreeSet+Sequence.swift"
  "TreeSet/TreeSet+SetAlgebra Initializers.swift"
  "TreeSet/TreeSet+SetAlgebra basics.swift"
  "TreeSet/TreeSet+SetAlgebra formIntersection.swift"
  "TreeSet/TreeSet+SetAlgebra formSymmetricDifference.swift"
  "TreeSet/TreeSet+SetAlgebra formUnion.swift"
  "TreeSet/TreeSet+SetAlgebra intersection.swift"
  "TreeSet/TreeSet+SetAlgebra isDisjoint.swift"
  "TreeSet/TreeSet+SetAlgebra isEqualSet.swift"
  "TreeSet/TreeSet+SetAlgebra isStrictSubset.swift"
  "TreeSet/TreeSet+SetAlgebra isStrictSuperset.swift"
  "TreeSet/TreeSet+SetAlgebra isSubset.swift"
  "TreeSet/TreeSet+SetAlgebra isSuperset.swift"
  "TreeSet/TreeSet+SetAlgebra subtract.swift"
  "TreeSet/TreeSet+SetAlgebra subtracting.swift"
  "TreeSet/TreeSet+SetAlgebra symmetricDifference.swift"
  "TreeSet/TreeSet+SetAlgebra union.swift"
  "TreeSet/TreeSet.swift"
  )
