/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.management;

import javax.management.MBeanServerConnection;
import javax.management.remote.JMXConnector;
import javax.management.remote.JMXConnectorFactory;
import javax.management.remote.JMXServiceURL;

/**
 * Test that verifies JMX connector server can be connected by
 * a client.
 *
 * @version $Revision: 680911 $
 *
 */
public class JmxInstrumentationWithConnectorTest extends JmxInstrumentationUsingDefaultsTest {

    protected static final String JMXSERVICEURL =
        "service:jmx:rmi:///jndi/rmi://localhost:2000/jmxrmi/camel";
    protected JMXConnector clientConnector;

    @Override
    protected void setUp() throws Exception {
        sleepForConnection = 2000;
        System.setProperty(JmxSystemPropertyKeys.CREATE_CONNECTOR, "True");
        // need to explicit set it to false to use non-platform mbs
        System.setProperty(JmxSystemPropertyKeys.USE_PLATFORM_MBS, "False");
        System.setProperty(JmxSystemPropertyKeys.REGISTRY_PORT, "2000");
        super.setUp();
    }

    @Override
    protected void tearDown() throws Exception {
        System.clearProperty(JmxSystemPropertyKeys.REGISTRY_PORT);
        System.clearProperty(JmxSystemPropertyKeys.CREATE_CONNECTOR);
        System.clearProperty(JmxSystemPropertyKeys.USE_PLATFORM_MBS);

        if (clientConnector != null) {
            try {
                clientConnector.close();
            } catch (Exception e) {
                // ignore
            }
            clientConnector = null;
        }
        super.tearDown();
    }

    @Override
    protected MBeanServerConnection getMBeanConnection() throws Exception {
        if (mbsc == null) {
            if (clientConnector == null) {
                clientConnector = JMXConnectorFactory.connect(
                        new JMXServiceURL(JMXSERVICEURL), null);
            }
            mbsc = clientConnector.getMBeanServerConnection();
        }
        return mbsc;
    }

}
