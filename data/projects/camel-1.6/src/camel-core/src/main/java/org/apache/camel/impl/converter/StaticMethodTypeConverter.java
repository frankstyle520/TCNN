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
package org.apache.camel.impl.converter;

import java.lang.reflect.Method;

import org.apache.camel.Exchange;
import org.apache.camel.TypeConverter;
import org.apache.camel.util.ObjectHelper;

/**
 * A {@link TypeConverter} implementation which invokes a static method to convert from a type to another type
 *
 * @version $Revision: 687545 $
 */
public class StaticMethodTypeConverter implements TypeConverter {
    private final Method method;
    private final boolean useExchange;

    public StaticMethodTypeConverter(Method method) {
        this.method = method;
        this.useExchange = method.getParameterTypes().length == 2;
    }

    @Override
    public String toString() {
        return "StaticMethodTypeConverter: " + method;
    }

    public <T> T convertTo(Class<T> type, Object value) {
        return convertTo(type, null, value);
    }

    public <T> T convertTo(Class<T> type, Exchange exchange, Object value) {
        return useExchange ? (T)ObjectHelper.invokeMethod(method, null, value, exchange)
            : (T)ObjectHelper.invokeMethod(method, null, value);
    }
}
