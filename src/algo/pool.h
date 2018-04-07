#pragma once

#include "defines.h"

class TPool {
public:
    TFeatures Features;
    TTarget Target;
    TNames Names;
    size_t FeatureCount = 0;
    size_t Size = 0;

public:
    void LoadFromFile(const std::string& name, bool train);
};

