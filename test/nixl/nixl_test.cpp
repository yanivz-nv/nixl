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
#include <iostream>
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"

constexpr int NUM_TRANSFERS = 1;
constexpr int BUFF_SIZE = 1024;
constexpr uint8_t BUFF_GOAL_VAL = 0xbb;

enum dlist_index {
    DLIST_DRAM_FOR_UCX1,
    DLIST_DRAM_FOR_UCX2,
    DLIST_NUM,
};

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

class DlistContext {
    nixl_reg_dlist_t dlist;
    std::vector<std::unique_ptr<uint8_t[]>> addrs;

public:
    DlistContext(nixl_mem_t type, size_t num_transfers, size_t buff_size, uint8_t init_val) :
        dlist(type),
        addrs(num_transfers) {

        assert(type == DRAM_SEG);

        for (size_t i = 0; i < num_transfers; i++) {
            addrs[i] = std::make_unique<uint8_t[]>(buff_size);
            std::fill_n(addrs[i].get(), buff_size, init_val);
            std::cout << "\tAllocated " << buff_size << " bytes at "
                      << (void *)addrs[i].get() << ", "
                      << "set to 0x" << std::hex << (int)init_val << std::dec
                      << std::endl;

            nixlBlobDesc blob_desc = {
                (uintptr_t)(addrs[i].get()),
                buff_size,
                0,
                nixl_blob_t(),
            };
            dlist.addDesc(blob_desc);
        }
    }

    DlistContext(const DlistContext &other) = delete;
    DlistContext& operator=(const DlistContext &other) = delete;

    DlistContext(DlistContext &&other) = default;
    DlistContext& operator=(DlistContext &&other) = default;

    ~DlistContext() = default;

    int waitForBuffersUpdate(uint8_t value) {
        bool res;

        for (size_t i = 0; i < addrs.size(); i++) {
            res = false;
            // TODO: timeout
            while (!res)
                res = allBytesAre(addrs[i].get(), BUFF_SIZE, value);
            std::cout << "   Buffer [" << i+1 << "/" << addrs.size() << "] completed\n";
        }
        if (!res) {
            std::cerr << "  Transfer failed, buffers are different\n";
            return -1;
        }

        std::cout << "  Transfer completed and Buffers match with Initiator\n";
        return 0;
    }

    bool allBytesAre(void *buffer, size_t size, uint8_t value) {
        uint8_t *byte_buffer = static_cast<uint8_t *>(buffer); // Cast void* to uint8_t*
        // Iterate over each byte in the buffer
        for (size_t i = 0; i < size; ++i) {
            if (byte_buffer[i] != value) {
                return false; // Return false if any byte doesn't match the value
            }
        }
        return true; // All bytes match the value
    }

    nixl_reg_dlist_t &getDlist() { return dlist; };
    const nixl_reg_dlist_t &getDlist() const { return dlist; };
};

class AgentTest {
private:
    // Member variables
    std::string role_;

    std::unique_ptr<nixlMDStreamListener> md_listener_;
    std::unique_ptr<nixlMDStreamClient> md_client_;

public:
    AgentTest(const std::string &role, const std::string &initiator_ip, int initiator_port);
    ~AgentTest();
    int testSingleTransfer();
    int testPartialMdUpdate();

private:
    std::vector<DlistContext> createDlistContexts(int dlist_num);
    std::string recvFromTarget();
    void sendToInitiator(const std::string &data);
    int initiateSingleTransfer(nixlAgent &agent, nixl_opt_args_t &extra_params, DlistContext &dlist_ctx);
    int testSingleTransferTargetFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts);
    int testSingleTransferInitiatorFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts);

    int testPartialMdUpdateTargetFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts);
    int testPartialMdUpdateInitiatorFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts);
};

AgentTest::AgentTest(const std::string &role, const std::string &initiator_ip, int initiator_port)
    : role_(role) {

    std::transform(role_.begin(), role_.end(), role_.begin(), ::tolower);

    if (!role_.compare("initiator") && !role_.compare("target")) {
        std::cerr << "Invalid role. Use 'initiator' or 'target'."
                << "Currently "<< role_ <<std::endl;
        throw std::invalid_argument("Invalid role");
    }

    if (role_ == "initiator") {
        md_listener_ = std::make_unique<nixlMDStreamListener>(initiator_port);
        md_listener_->startListenerForClient();
    } else {
        md_client_ = std::make_unique<nixlMDStreamClient>(initiator_ip, initiator_port);
        while (!md_client_->connectListener()) {
            std::cout << "Waiting for listener to connect...\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

AgentTest::~AgentTest() {
}

std::vector<DlistContext>
AgentTest::createDlistContexts(int dlist_num) {
    uint8_t init_val = role_ == "target" ? 0 : BUFF_GOAL_VAL;
    std::vector<DlistContext> dlist_contexts;

    for (int i = 0; i < dlist_num; i++) {
        std::cout << "Creating dlist context for " << role_ << " :";
        dlist_contexts.emplace_back(DRAM_SEG, NUM_TRANSFERS, BUFF_SIZE, init_val);
    }
    return dlist_contexts;
}

std::string AgentTest::recvFromTarget() {
    return md_listener_->recvFromClient();
}

void
AgentTest::sendToInitiator(const std::string &data) {
    md_client_->sendData(data);
}

int
AgentTest::initiateSingleTransfer(nixlAgent &agent, nixl_opt_args_t &extra_params, DlistContext &dlist_ctx) {
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes remote_serdes;
    nixl_blob_t tgt_md_init;
    nixl_status_t status;
    nixlXferReqH* treq;

    std::cout << " Receive metadata from Target \n";
    std::cout << " \t -- To be handled by runtime - currently received via a TCP Stream\n";
    std::string rrstr = recvFromTarget();

    remote_serdes.importStr(rrstr);
    tgt_md_init = remote_serdes.getStr("AgentMD");
    assert (tgt_md_init != "");
    std::string target_name;
    status = agent.loadRemoteMD(tgt_md_init, target_name);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Error loading remote metadata\n";
        return -1;
    }

    std::cout << " Verify Deserialized Target's Desc List at Initiator\n";
    nixl_xfer_dlist_t dram_target_ucx(&remote_serdes);
    nixl_xfer_dlist_t dram_initiator_ucx = dlist_ctx.getDlist().trim();
    dram_target_ucx.print();

    std::cout << " Got metadata from " << target_name << " \n";
    std::cout << " End Control Path metadata exchanges \n";
    std::cout << " Start Data Path Exchanges \n\n";
    std::cout << " Create transfer request with UCX backend\n ";

    status = agent.createXferReq(NIXL_WRITE, dram_initiator_ucx, dram_target_ucx,
                            "target", treq, &extra_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "Error creating transfer request\n";
        return -1;
    }

    std::cout << " Post the request with UCX backend\n ";
    status = agent.postXferReq(treq);
    std::cout << " Initiator posted Data Path transfer\n";
    std::cout << " Waiting for completion\n";

    while (status != NIXL_SUCCESS) {
        status = agent.getXferStatus(treq);
        assert(status >= 0);
    }
    std::cout << " Completed Sending Data using UCX backend\n";
    agent.releaseXferReq(treq);

    return 0;
}

int
AgentTest::testSingleTransfer() {
    /** NIXL declarations */
    /** Agent and backend creation parameters */
    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBackendH* ucx;
    int rc;

    std::cout << std::endl;
    std::cout << "========================================\n";
    std::cout << "Starting Test: Single Transfer\n";
    std::cout << "========================================\n";

    constexpr int dlist_num = 1;
    static_assert(DLIST_DRAM_FOR_UCX1 < dlist_num, "dlist_num must include DLIST_DRAM_FOR_UCX1");

    /** List of dlist contexts for each transfer */
    std::vector<DlistContext> dlist_contexts = createDlistContexts(dlist_num);

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for "<< role_ << "\n";
    nixlAgent agent(role_, cfg);
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    /** Register memory in both initiator and target */
    for (int i = 0; i < dlist_num; i++)
        agent.registerMem(dlist_contexts[i].getDlist(), &extra_params);

    std::cout << " Start Control Path metadata exchanges \n";

    if (role_ == "target")
        rc = testSingleTransferTargetFlow(agent, extra_params, dlist_contexts);
    else
        rc = testSingleTransferInitiatorFlow(agent, extra_params, dlist_contexts);

    std::cout <<"Cleanup.. \n";
    for (int i = 0; i < dlist_num; i++)
        agent.deregisterMem(dlist_contexts[i].getDlist(), &extra_params);

    return rc;
}

int
AgentTest::testSingleTransferTargetFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts) {
    nixl_blob_t tgt_metadata;
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes serdes;
    int rc;

    agent.getLocalMD(tgt_metadata);

    enum dlist_index dlist_idx = DLIST_DRAM_FOR_UCX1;
    DlistContext *dlist_ctx = &dlist_contexts[dlist_idx];

    std::cout << " Desc List from Target to Initiator\n";
    dlist_ctx->getDlist().print();

    /** Sending both metadata strings together */
    assert(serdes.addStr("AgentMD", tgt_metadata) == NIXL_SUCCESS);
    assert(dlist_ctx->getDlist().trim().serialize(&serdes) == NIXL_SUCCESS);

    std::cout << " Serialize Metadata to string and Send to Initiator\n";
    std::cout << " \t -- To be handled by runtime - currently sent via a TCP Stream\n";
    sendToInitiator(serdes.exportStr());
    std::cout << " End Control Path metadata exchanges \n";

    std::cout << " Start Data Path Exchanges \n";
    std::cout << " Waiting to receive Data from Initiator\n";

    std::cout << " Waiting UCX Transfer idx "<< dlist_idx << "\n";
    rc = dlist_ctx->waitForBuffersUpdate(BUFF_GOAL_VAL);
    if (rc != 0) {
        std::cerr << " UCX Transfer failed\n";
        return rc;
    }
    std::cout <<" UCX Transfer Success!!!\n";
    return rc;
}

int
AgentTest::testSingleTransferInitiatorFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts) {
    return initiateSingleTransfer(agent, extra_params, dlist_contexts[DLIST_DRAM_FOR_UCX1]);
}

int
AgentTest::testPartialMdUpdate() {
    /** NIXL declarations */
    /** Agent and backend creation parameters */
    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBackendH* ucx;
    int rc;

    std::cout << std::endl;
    std::cout << "========================================\n";
    std::cout << "Starting Test: Partial Metadata Update\n";
    std::cout << "========================================\n";

    constexpr int dlist_num = 2;
    static_assert(DLIST_DRAM_FOR_UCX1 < dlist_num, "dlist_num must include DLIST_DRAM_FOR_UCX1");
    static_assert(DLIST_DRAM_FOR_UCX2 < dlist_num, "dlist_num must include DLIST_DRAM_FOR_UCX2");

    /** List of dlist contexts for each transfer */
    std::vector<DlistContext> dlist_contexts = createDlistContexts(dlist_num);

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for "<< role_ << "\n";
    nixlAgent agent(role_, cfg);
    agent.createBackend("UCX", params, ucx);

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(ucx);

    /** Register memory in both initiator and target */
    for (int i = 0; i < dlist_num; i++)
        agent.registerMem(dlist_contexts[i].getDlist(), &extra_params);

    std::cout << " Start Control Path metadata exchanges \n";

    if (role_ == "target")
        rc = testPartialMdUpdateTargetFlow(agent, extra_params, dlist_contexts);
    else
        rc = testPartialMdUpdateInitiatorFlow(agent, extra_params, dlist_contexts);

    std::cout <<"Cleanup.. \n";
    for (int i = 0; i < dlist_num; i++)
        agent.deregisterMem(dlist_contexts[i].getDlist(), &extra_params);

    return rc;
}

int
AgentTest::testPartialMdUpdateTargetFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts) {
    nixl_blob_t tgt_metadata;
    int rc;

    for (size_t dlist_idx = 0; dlist_idx < dlist_contexts.size(); dlist_idx++) {
        std::cout << "----------------------------------------\n";
        std::cout << "Metadata for first transfer:\n";
        std::cout << " Desc List from Target to Initiator\n";

        /** Serialization/Deserialization object to create a blob */
        nixlSerDes serdes;
        DlistContext *dlist_ctx = &dlist_contexts[dlist_idx];

        agent.getLocalPartialMD(dlist_ctx->getDlist(), true, tgt_metadata, &extra_params);
        dlist_ctx->getDlist().print();

        /** Sending both metadata strings together */
        assert(serdes.addStr("AgentMD", tgt_metadata) == NIXL_SUCCESS);
        assert(dlist_ctx->getDlist().trim().serialize(&serdes) == NIXL_SUCCESS);

        std::cout << " Serialize Metadata to string and Send to Initiator\n";
        std::cout << " \t -- To be handled by runtime - currently sent via a TCP Stream\n";
        sendToInitiator(serdes.exportStr());
        std::cout << " End Control Path metadata exchanges \n";

        std::cout << " Start Data Path Exchanges \n";
        std::cout << " Waiting to receive Data from Initiator\n";

        std::cout << " Waiting UCX Transfer idx "<< dlist_idx << "\n";
        rc = dlist_ctx->waitForBuffersUpdate(BUFF_GOAL_VAL);
        if (rc != 0) {
            std::cerr << " UCX Transfer [" << dlist_idx+1 << "/" << dlist_contexts.size() << "] failed\n";
            return rc;
        }
        std::cout <<" UCX Transfer [" << dlist_idx+1 << "/" << dlist_contexts.size() << "] Success!!!\n";
    }

    return rc;
}

int
AgentTest::testPartialMdUpdateInitiatorFlow(nixlAgent &agent, nixl_opt_args_t &extra_params, std::vector<DlistContext> &dlist_contexts) {
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes remote_serdes;
    nixl_blob_t tgt_md_init;
    int rc;

    for (size_t i = 0; i < dlist_contexts.size(); i++) {
        std::cout << "----------------------------------------\n";
        rc = initiateSingleTransfer(agent, extra_params, dlist_contexts[i]);
        if (rc != 0) {
            std::cerr << "Transfer [" << i+1 << "/" << dlist_contexts.size() << "] failed\n";
            return rc;
        }
        std::cout << "Transfer [" << i+1 << "/" << dlist_contexts.size() << "] completed\n";
    }

    return 0;
}

int
main(int argc, char *argv[]) {
    /** Argument Parsing */
    if (argc < 4) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> " <<"<Initiator IP> <Initiator Port>"
                << std::endl;
        return -1;
    }

    std::string role = argv[1];
    std::string initiator_ip = argv[2];
    int initiator_port = std::stoi(argv[3]);
    int rc;

    try {
        AgentTest test(role, initiator_ip, initiator_port);
        std::string test_name = "Single Transfer";
        rc = test.testSingleTransfer();
        if (rc != 0) {
            std::cerr << "Test [" << test_name << "] failed\n";
            return rc;
        }
        std::cout << "Test [" << test_name << "] success\n";
        std::cout << "========================================\n";

        test_name = "Partial MD Update";
        rc = test.testPartialMdUpdate();
        if (rc != 0) {
            std::cerr << "Test [" << test_name << "] failed\n";
            return rc;
        }
        std::cout << "Test [" << test_name << "] success\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return rc;
}
