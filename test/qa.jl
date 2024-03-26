using Aqua
using WeightInitializers

Aqua.test_all(WeightInitializers; ambiguities=false)
Aqua.test_ambiguities(WeightInitializers; recursive=false)