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
package org.apache.camel.processor;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.processor.interceptor.Delayer;

/**
 * Delay interceptor unit test.
 *
 * @version $Revision: 711235 $
 */
public class DelayInterceptorTest extends ContextTestSupport {

    public void testSendingSomeMessages() throws Exception {
        long start = System.currentTimeMillis();
        for (int i = 0; i < 10; i++) {
            template.sendBody("direct:start", "Message #" + i);
        }
        long delta = System.currentTimeMillis() - start;
        assertTrue("Should be slower to run: " + delta, delta > 4000);
        assertTrue("Should not take that long to run: " + delta, delta < 7000);
    }

    @Override
    protected void setUp() throws Exception {
        disableJMX();
        super.setUp();
    }

    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            // START SNIPPET: e1
            public void configure() throws Exception {
                // add the delay interceptor to delay each step 200 millis
                getContext().addInterceptStrategy(new Delayer(200));
                
                // regular routes here
            // END SNIPPET: e1

                from("direct:start").
                        process(new Processor() {
                            public void process(Exchange exchange) throws Exception {
                                // do nothing
                            }
                        }).
                        to("mock:result");
            }
        };
    }

}
