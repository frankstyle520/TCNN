/**
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.jmxconnect.provider.activemq;


import org.apache.camel.jmxconnect.CamelJmxConnectorServer;
import org.apache.camel.jmxconnect.CamelJmxConnectorSupport;
import org.apache.camel.CamelContext;

import javax.management.MBeanServer;
import javax.management.remote.JMXConnectorServer;
import javax.management.remote.JMXConnectorServerProvider;
import javax.management.remote.JMXServiceURL;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.Map;

/**
 * @version $Revision: 689344 $
 */
public class ServerProvider implements JMXConnectorServerProvider {

    public JMXConnectorServer newJMXConnectorServer(JMXServiceURL url, Map environment, MBeanServer server) throws IOException {
        String brokerUrl = CamelJmxConnectorSupport.getEndpointUri(url, "activemq");
        CamelJmxConnectorServer answer = new CamelJmxConnectorServer(url, ActiveMQHelper.getDefaultEndpointUri(), environment, server);

        // now lets configure the ActiveMQ component
        CamelContext camelContext = answer.getCamelContext();
        ActiveMQHelper.configureActiveMQComponent(camelContext, brokerUrl);
        return answer;
    }
}
