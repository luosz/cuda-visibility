#pragma once

#ifndef SERIALIZE_H
#define SERIALIZE_H

template<class Archive>
void serialize(Archive &archive, float4 &m)
{
	archive(cereal::make_nvp("r", m.x), cereal::make_nvp("g", m.y), cereal::make_nvp("b", m.z), cereal::make_nvp("a", m.w));
}

template<class Archive>
void serialize(Archive &archive, float3 &m)
{
	archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y), cereal::make_nvp("z", m.z));
}

template<class Archive>
void serialize(Archive &archive, int2 &m)
{
	archive(cereal::make_nvp("x", m.x), cereal::make_nvp("y", m.y));
}

#endif // SERIALIZE_H
