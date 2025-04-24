/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __MEM_SECTION_H
#define __MEM_SECTION_H

#include <vector>
#include <unordered_map>
#include <map>
#include <array>
#include <string>
#include <set>
#include "nixl_descriptors.h"
#include "nixl.h"
#include "backend/backend_engine.h"

typedef std::pair<nixl_mem_t, nixlBackendEngine*>              section_key_t;
typedef std::set<nixlBackendEngine*>                           backend_set_t;
typedef std::unordered_map<nixl_backend_t, nixlBackendEngine*> backend_map_t;

/**
 * @brief Section descriptor for nixl
 *
 * This class is used to store a section descriptor for nixl.
 * It is derived from nixlMetaDesc and contains the meta blob for the section.
 */
class nixlSectionDesc : public nixlMetaDesc {
public:
    nixl_blob_t metaBlob;

    using nixlMetaDesc::nixlMetaDesc;

    nixl_blob_t serialize() const {
        // Serialize only the meta blob. metadataP is private so we don't include it.
        // The other side will deserialize it as nixlBlobDesc.
        return nixlBasicDesc::serialize() + metaBlob;
    }

    inline friend bool operator==(const nixlSectionDesc &lhs, const nixlSectionDesc &rhs) {
        return (static_cast<nixlMetaDesc>(lhs) == static_cast<nixlMetaDesc>(rhs));
    }

    inline void print(const std::string &suffix) const {
        nixlMetaDesc::print(", meta blob: " + metaBlob + suffix);
    }
};

typedef nixlDescList<nixlSectionDesc>               nixl_sec_dlist_t;
typedef std::map<section_key_t, nixl_sec_dlist_t*>  section_map_t;

class nixlMemSection {
    protected:
        std::array<backend_set_t, FILE_SEG+1>         memToBackend;
        section_map_t                                 sectionMap;

    public:
        nixlMemSection () {};

        backend_set_t* queryBackends (const nixl_mem_t &mem);

        nixl_status_t populate (const nixl_xfer_dlist_t &query,
                                nixlBackendEngine* backend,
                                nixl_meta_dlist_t &resp) const;


        virtual ~nixlMemSection () = 0; // Making the class abstract
};


class nixlLocalSection : public nixlMemSection {
    public:
        nixl_status_t addDescList (const nixl_reg_dlist_t &mem_elms,
                                   nixlBackendEngine* backend,
                                   nixl_sec_dlist_t &remote_self);

        // Each nixlBasicDesc should be same as original registration region
        nixl_status_t remDescList (const nixl_reg_dlist_t &mem_elms,
                                   nixlBackendEngine* backend);

        nixl_status_t serialize(nixlSerDes* serializer) const;

        nixl_status_t serializePartial(nixlSerDes* serializer,
                                       const backend_set_t &backends,
                                       const nixl_reg_dlist_t &mem_elms) const;

        ~nixlLocalSection();
};


class nixlRemoteSection : public nixlMemSection {
    private:
        std::string agentName;

        nixl_status_t addDescList (
                           const nixl_reg_dlist_t &mem_elms,
                           nixlBackendEngine *backend);
    public:
        nixlRemoteSection (const std::string &agent_name);

        nixl_status_t loadRemoteData (nixlSerDes* deserializer,
                                      backend_map_t &backendToEngineMap);

        // When adding self as a remote agent for local operations
        nixl_status_t loadLocalData (const nixl_sec_dlist_t& mem_elms,
                                     nixlBackendEngine* backend);
        ~nixlRemoteSection();
};

#endif
